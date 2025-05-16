import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from nltk import sent_tokenize
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id):
    processed_id = cur_sess_id.replace("answer_", "").replace("noans_", "")
    # candidate_results = [v for k, v in index_expansion_result_cache.items() if processed_id in k]
    try:
        cur_expansion = index_expansion_result_cache[processed_id]
    except:
        # failure to generate the expansion; index_expansion_result_cache could also contain None values.
        cur_expansion = None
    if type(cur_expansion) == str:
        cur_expansion = [cur_expansion]
    return cur_expansion


def resolve_expansion(
    expansion_type,
    resolution_strategy,
    existing_corpus,
    existing_corpus_ids,
    existing_corpus_timestamps,
    cur_item_expansions,
    cur_sess_id,
    ts,
):
    # preprocess and split expansion, if applicable
    if expansion_type == "session-summ":
        # print(cur_item_expansions)
        if cur_item_expansions is None:
            cur_item_expansions = [""]
        assert len(cur_item_expansions) == 1
        if "split" in resolution_strategy:
            cur_item_expansions = sent_tokenize(cur_item_expansions[0])
    elif expansion_type == "session-keyphrase" or expansion_type == "turn-keyphrase":
        if cur_item_expansions is None:
            cur_item_expansions = [""]
            # print('Warning: none value for keyphrase expansion')
        assert len(cur_item_expansions) == 1
        if "split" in resolution_strategy:
            cur_item_expansions = [x.strip() for x in cur_item_expansions[0].split(";")]
    elif expansion_type == "session-userfact" or expansion_type == "turn-userfact":
        if cur_item_expansions is None:
            # For failed expansions, we treat it as if the expansion is an empty string
            cur_item_expansions = [""]
        cur_item_expansions = [str(x) for x in cur_item_expansions]
        if "split" not in resolution_strategy:
            if cur_item_expansions:
                cur_item_expansions = [" ".join(cur_item_expansions)]
            else:
                cur_item_expansions = []
    else:
        raise NotImplementedError

    # merge expansion with the main items
    if "separate" in resolution_strategy:
        existing_corpus += [str(x) for x in cur_item_expansions]
        existing_corpus_ids += [cur_sess_id for _ in cur_item_expansions]
        existing_corpus_timestamps += [ts for _ in cur_item_expansions]
    elif "merge" in resolution_strategy or "replace" in resolution_strategy:
        out_corpus, out_corpus_ids, out_corpus_timestamps = [], [], []
        N = len(existing_corpus_ids)
        for i in range(N):
            if existing_corpus_ids[i] == cur_sess_id:
                if "merge" in resolution_strategy:
                    for expansion_item in cur_item_expansions:
                        out_corpus.append(expansion_item + " " + existing_corpus[i])
                        # print(existing_corpus[i], '--->', expansion_item + ' ' + existing_corpus[i])
                        # print('\n\n+++\n\n')
                        out_corpus_ids.append(existing_corpus_ids[i])
                        out_corpus_timestamps.append(existing_corpus_timestamps[i])
                elif "replace" in resolution_strategy:
                    for expansion_item in cur_item_expansions:
                        out_corpus.append(expansion_item)  # different
                        out_corpus_ids.append(existing_corpus_ids[i])
                        out_corpus_timestamps.append(existing_corpus_timestamps[i])
                else:
                    raise NotImplementedError
            else:
                out_corpus.append(existing_corpus[i])
                out_corpus_ids.append(existing_corpus_ids[i])
                out_corpus_timestamps.append(existing_corpus_timestamps[i])
        existing_corpus, existing_corpus_ids, existing_corpus_timestamps = (
            out_corpus,
            out_corpus_ids,
            out_corpus_timestamps,
        )
    else:
        raise NotImplementedError

    return existing_corpus, existing_corpus_ids, existing_corpus_timestamps


client = OpenAI(
    api_key="empty",
    base_url="http://localhost:8001/v1",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--outfile_prefix", type=str, default=None, required=False)
    parser.add_argument("--cache_dir")

    # basic parameters
    parser.add_argument(
        "--retriever",
        type=str,
        default="flat-contriever",
        choices=["oracle", "flat-bm25", "flat-contriever", "flat-stella", "flat-gte"],
    )
    parser.add_argument(
        "--granularity", type=str, default="session", choices=["session", "turn"]
    )

    # index expansion
    parser.add_argument(
        "--index_expansion_method",
        type=str,
        default="none",
        choices=[
            "none",
            "session-summ",
            "session-keyphrase",
            "session-userfact",
            "turn-keyphrase",
            "turn-userfact",
        ],
    )
    parser.add_argument("--index_expansion_llm", type=str, default="none")
    parser.add_argument("--index_expansion_result_cache", type=str, default=None)
    parser.add_argument(
        "--index_expansion_result_join_mode",
        type=str,
        default="none",
        choices=[
            "separate",
            "split-separate",
            "merge",
            "split-merge",
            "replace",
            "split-replace",
            "none",
        ],
    )
    return parser.parse_args()


def check_args(args):
    # print(args)
    if args.index_expansion_method != "none":
        print(
            "Note: index expansion method {} specified".format(
                args.index_expansion_method
            )
        )
        assert (
            args.index_expansion_result_join_mode is not None
            and args.index_expansion_result_join_mode != "none"
        )
        if args.index_expansion_method in [
            "session-summ",
            "session-keyphrase",
            "session_userfact",
        ]:
            assert args.granularity == "session"
        if (
            args.index_expansion_result_cache is not None
            and args.index_expansion_result_cache != None
        ):
            assert args.index_expansion_method in args.index_expansion_result_cache
            print(
                "Using cached index expansion results at",
                args.index_expansion_result_cache,
            )


def get_outfile_prefix(args):
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != "none":
        outfile_prefix = args.outfile_prefix
    else:
        outfile_prefix = args.in_file.split("/")[-1]
    return outfile_prefix


class DenseRetrievalMaster:
    def __init__(self, retriever, gpu_id):
        self.retriever = retriever
        self.device = torch.device("cuda", gpu_id)  # @Mufan: use later 4 gpus
        # print('Initializing DenseRetrievalMaster with device', self.device)
        self.prepare_retriever()
        self.corpus_cache = {}

    def prepare_retriever(self):
        self.retriever_model = None

        if self.retriever == "flat-contriever":
            print(self.device)
            model = AutoModel.from_pretrained("facebook/contriever").to(self.device)
            tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            self.retriever_model = (tokenizer, model)

        elif self.retriever == "flat-stella":
            model_dir = (
                "/data/mufan/agent/LongMemEval/model_cache"
                + "/dunzhang_stella_en_1.5B_v5"
            )
            vector_dim = 1024
            vector_linear_directory = f"2_Dense_{vector_dim}"
            model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(
                self.device
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            vector_linear = torch.nn.Linear(
                in_features=model.config.hidden_size, out_features=vector_dim
            ).to(self.device)
            vector_linear_dict = {
                k.replace("linear.", ""): v
                for k, v in torch.load(
                    os.path.join(
                        model_dir, f"{vector_linear_directory}/pytorch_model.bin"
                    )
                ).items()
            }
            vector_linear.load_state_dict(vector_linear_dict)
            vector_linear.to(self.device)
            self.retriever_model = (tokenizer, model, vector_linear)

        elif self.retriever == "flat-gte":
            tokenizer = AutoTokenizer.from_pretrained(
                "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True
            ).to(self.device)
            model.eval()
            self.retriever_model = (tokenizer, model)

    def run_flat_retrieval(self, query, retriever, corpus, corpus_key=None):
        if retriever == "flat-bm25":
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            # tokenized_torpus = word_tokenize(corpus)
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.split(" "))
            return np.argsort(scores)[::-1]

        elif retriever in ["flat-contriever", "flat-stella", "flat-gte"]:
            model2bsz = {"flat-contriever": 128, "flat-stella": 64, "flat-gte": 1}
            bsz = model2bsz[retriever]

            if retriever == "flat-contriever":
                tokenizer, model = self.retriever_model

                def mean_pooling(token_embeddings, mask):
                    token_embeddings = token_embeddings.masked_fill(
                        ~mask[..., None].bool(), 0.0
                    )
                    sentence_embeddings = (
                        token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                    )
                    return sentence_embeddings

                with torch.no_grad():
                    inputs = tokenizer(
                        [query], padding=True, truncation=True, return_tensors="pt"
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    query_vectors = (
                        mean_pooling(outputs[0], inputs["attention_mask"])
                        .detach()
                        .cpu()
                    )
                    if corpus_key in self.corpus_cache:
                        all_docs_vectors = self.corpus_cache[corpus_key]
                    elif os.path.exists(corpus_key + ".npy"):
                        all_docs_vectors = np.load(corpus_key + ".npy")
                        self.corpus_cache[corpus_key] = all_docs_vectors
                    else:
                        print("Caching corpus vectors for key", corpus_key)
                        all_docs_vectors = []
                        dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                        for batch in dataloader:
                            inputs = tokenizer(
                                batch,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                            )
                            inputs = {k: v.to(model.device) for k, v in inputs.items()}
                            outputs = model(**inputs)
                            cur_docs_vectors = (
                                mean_pooling(outputs[0], inputs["attention_mask"])
                                .detach()
                                .cpu()
                            )
                            all_docs_vectors.append(cur_docs_vectors)
                        all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                        np.save(corpus_key + ".npy", all_docs_vectors)
                        self.corpus_cache[corpus_key] = all_docs_vectors

                    scores = (query_vectors @ all_docs_vectors.T).squeeze()

            elif retriever == "flat-stella":
                tokenizer, model, vector_linear = self.retriever_model
                with torch.no_grad():
                    input_data = tokenizer(
                        [query],
                        padding="longest",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    input_data = {k: v.to(model.device) for k, v in input_data.items()}
                    attention_mask = input_data["attention_mask"]
                    last_hidden_state = model(**input_data)[0]
                    last_hidden = last_hidden_state.masked_fill(
                        ~attention_mask[..., None].bool(), 0.0
                    )
                    query_vectors = (
                        last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                    )
                    query_vectors = normalize(
                        vector_linear(query_vectors).detach().cpu()
                    )
                with torch.no_grad():
                    all_docs_vectors = []
                    dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        input_data = tokenizer(
                            batch,
                            padding="longest",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        )
                        input_data = {
                            k: v.to(model.device) for k, v in input_data.items()
                        }
                        attention_mask = input_data["attention_mask"]
                        last_hidden_state = model(**input_data)[0]
                        last_hidden = last_hidden_state.masked_fill(
                            ~attention_mask[..., None].bool(), 0.0
                        )
                        docs_vectors = (
                            last_hidden.sum(dim=1)
                            / attention_mask.sum(dim=1)[..., None]
                        )
                        docs_vectors = normalize(
                            vector_linear(docs_vectors).detach().cpu()
                        )
                        all_docs_vectors.append(docs_vectors)
                    all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                scores = torch.tensor((query_vectors @ all_docs_vectors.T).squeeze())

            elif retriever == "flat-gte":

                def last_token_pool(
                    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
                ) -> torch.Tensor:
                    left_padding = (
                        attention_mask[:, -1].sum() == attention_mask.shape[0]
                    )
                    if left_padding:
                        return last_hidden_states[:, -1]
                    else:
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = last_hidden_states.shape[0]
                        return last_hidden_states[
                            torch.arange(batch_size, device=last_hidden_states.device),
                            sequence_lengths,
                        ]

                def get_detailed_instruct(task_description: str, query: str) -> str:
                    return f"Instruction: {task_description}\nQuery: {query}"

                tokenizer, model = self.retriever_model
                task = "Given a query about personal information, retrieve relevant chat history that answer the query."
                with torch.no_grad():
                    all_vectors = []
                    dataloader = DataLoader(
                        [get_detailed_instruct(task, query)] + corpus,
                        batch_size=bsz,
                        shuffle=False,
                    )
                    for batch in dataloader:
                        batch_dict = tokenizer(
                            batch,
                            max_length=8192,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )
                        batch_dict = {
                            k: v.to(model.device) for k, v in batch_dict.items()
                        }
                        outputs = model(**batch_dict)
                        embeddings = last_token_pool(
                            outputs.last_hidden_state, batch_dict["attention_mask"]
                        )
                        all_vectors.append(embeddings)
                all_vectors = torch.cat(all_vectors, dim=0)
                all_vectors = F.normalize(all_vectors, p=2, dim=1)
                scores = (all_vectors[:1] @ all_vectors[1:].T).squeeze()

            else:
                raise NotImplementedError

            return scores.argsort(descending=True)

        else:
            raise NotImplementedError


def process_item_flat_index(data, granularity, sess_id, timestamp):
    corpus = []

    if granularity == "session":
        text = " ".join(
            [interact["content"] for interact in data if interact["role"] == "user"]
        )
        corpus.append(text)
        ids = [sess_id]
        if "answer" in sess_id and all(
            [
                not turn["has_answer"]
                for turn in [x for x in data if x["role"] == "user"]
            ]
        ):
            ids = [sess_id.replace("answer", "noans")]
    elif granularity == "turn":
        ids = []
        for i_turn, turn in enumerate(data):
            if turn["role"] == "user":
                corpus.append(turn["content"])
                if "answer" not in sess_id:
                    ids.append(sess_id + "_" + str(i_turn + 1))
                else:
                    assert "has_answer" in turn
                    assert turn["has_answer"] in [True, False]
                    if turn["has_answer"]:
                        ids.append(sess_id + "_" + str(i_turn + 1))
                    else:
                        ids.append(
                            (sess_id + "_" + str(i_turn + 1)).replace("answer", "noans")
                        )
                        assert "answer" not in ids[-1]
    else:
        raise NotImplementedError

    return corpus, ids, [timestamp for _ in corpus]


from functools import cache


@cache
def get_retriever(retriever, gpu_id):
    return DenseRetrievalMaster(retriever, gpu_id)


def batch_get_retrieved_context_and_eval(
    entry_list,
    args,
    index_expansion_result_cache=None,
    k=5,
):
    gpu_id = 0
    if args.retriever in [
        "flat-bm25",
        "flat-contriever",
        "flat-stella",
        "flat-gte",
        "oracle",
    ]:
        retriever_master = get_retriever(args.retriever, gpu_id)
    else:
        raise NotImplementedError

    results = []
    for entry in tqdm(entry_list):
        # step 1: prepare corpus index (with potential index expansion)
        corpus, corpus_ids, corpus_timestamps = [], [], []
        for cur_sess_id, sess_entry, ts in zip(
            entry["haystack_session_ids"],
            entry["haystack_sessions"],
            entry["haystack_dates"],
        ):
            cur_items, cur_ids, cur_ts = process_item_flat_index(
                sess_entry, args.granularity, cur_sess_id, ts
            )
            corpus += cur_items
            corpus_ids += cur_ids
            corpus_timestamps += cur_ts

        if args.index_expansion_method != "none":
            if index_expansion_result_cache is not None:
                if "session" in args.index_expansion_method:
                    for cur_sess_id, sess_entry, ts in zip(
                        entry["haystack_session_ids"],
                        entry["haystack_sessions"],
                        entry["haystack_dates"],
                    ):
                        cur_item_expansions = fetch_expansion_from_cache(
                            index_expansion_result_cache, cur_sess_id
                        )
                        # print(cur_sess_id)
                        # print(cur_item_expansions)
                        corpus, corpus_ids, corpus_timestamps = resolve_expansion(
                            args.index_expansion_method,
                            args.index_expansion_result_join_mode,
                            corpus,
                            corpus_ids,
                            corpus_timestamps,
                            cur_item_expansions,
                            cur_sess_id,
                            ts,
                        )
                elif "turn" in args.index_expansion_method:
                    for cur_sess_id, sess_entry, ts in zip(
                        entry["haystack_session_ids"],
                        entry["haystack_sessions"],
                        entry["haystack_dates"],
                    ):
                        for cur_turn_id, cur_turn_content in enumerate(sess_entry):
                            if cur_turn_content["role"] == "user":
                                cur_item_expansions = fetch_expansion_from_cache(
                                    index_expansion_result_cache,
                                    cur_sess_id + f"_{cur_turn_id+1}",
                                )
                                corpus, corpus_ids, corpus_timestamps = (
                                    resolve_expansion(
                                        args.index_expansion_method,
                                        args.index_expansion_result_join_mode,
                                        corpus,
                                        corpus_ids,
                                        corpus_timestamps,
                                        cur_item_expansions,
                                        cur_sess_id + f"_{cur_turn_id+1}",
                                        ts,
                                    )
                                )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        correct_docs = list(
            set([doc_id for doc_id in corpus_ids if "answer" in doc_id])
        )

        # step 2: run retrieval
        # query = entry["question"]
        query = args.query

        if args.retriever in [
            "flat-bm25",
            "flat-contriever",
            "flat-stella",
            "flat-gte",
        ]:
            rankings = retriever_master.run_flat_retrieval(
                query, args.retriever, corpus, corpus_key=args.in_file
            )
        elif args.retriever in ["oracle"]:
            correct_idx, incorrect_idx = [], []
            for i_doc, cid in enumerate(corpus_ids):
                if cid in correct_docs:
                    correct_idx.append(i_doc)
                else:
                    incorrect_idx.append(i_doc)
            rankings = correct_idx + incorrect_idx
        else:
            raise NotImplementedError

        # step 3: record evaluation metrics
        cur_results = {
            # 'question_id': entry['question_id'],
            # 'question_type': entry['question_type'],
            # 'question': entry['question'],
            # 'answer': entry['answer'],
            # 'question_date': entry['question_date'],
            # 'haystack_dates': entry['haystack_dates'],
            # 'haystack_sessions': entry['haystack_sessions'],
            # 'haystack_session_ids': entry['haystack_session_ids'],
            # 'answer_session_ids': entry['answer_session_ids'],
            "retrieval_results": {
                "ranked_items": [
                    {
                        "corpus_id": corpus_ids[rid],
                        "text": corpus[rid],
                        # 'timestamp': corpus_timestamps[rid]
                    }
                    # for rid in rankings
                    for rid in rankings[:k]
                ],
                "metrics": {"session": {}, "turn": {}},
            }
        }
        # for k in [1, 3, 5, 10, 30, 50]:
        #     recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        #     cur_results['retrieval_results']['metrics'][args.granularity].update({
        #         'recall_any@{}'.format(k): recall_any,
        #         'recall_all@{}'.format(k): recall_all,
        #         'ndcg_any@{}'.format(k): ndcg_any
        #     })
        #     if args.granularity == 'turn':
        #         recall_any, recall_all, ndcg_any = evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=k)
        #         cur_results['retrieval_results']['metrics']['session'].update({
        #             'recall_any@{}'.format(k): recall_any,
        #             'recall_all@{}'.format(k): recall_all,
        #             'ndcg_any@{}'.format(k): ndcg_any
        #         })

        results.append(cur_results)

    return results


def retrieval(args):
    check_args(args)

    # outfile_prefix = get_outfile_prefix(args)
    # out_file = args.out_dir + '/' + outfile_prefix + '_retrievallog_{}_{}'.format(args.granularity, args.retriever)
    # # log_file = out_file + '.log'
    # # log_f, out_f = open(log_file, 'w'), open(out_file, 'w')
    # out_f = open(out_file, 'w')

    # load data and cache
    in_data = json.load(open(args.in_file))
    n_has_abstention = len([x for x in in_data if "_abs" in x["question_id"]])
    if n_has_abstention > 0:
        print(
            "Warning: found {} abstention instances within the data".format(
                n_has_abstention
            )
        )

    index_expansion_result_cache = None
    if (
        args.index_expansion_result_cache is not None
        and args.index_expansion_result_cache != "none"
    ):
        index_expansion_result_cache = json.load(
            open(args.index_expansion_result_cache)
        )
        print("Loaded pre-computed expansions from", args.index_expansion_result_cache)

    # multiprocessing
    # num_processes = torch.cuda.device_count()
    num_processes = 1
    if "bm25" in args.retriever:
        num_processes = 10
    print("Setting num processes = {} with retriever {}".format(1, args.retriever))

    worker = partial(
        batch_get_retrieved_context_and_eval,
        args=args,
        index_expansion_result_cache=index_expansion_result_cache,
        k=args.k,
    )

    results = worker(in_data)

    # for cur_results in results:
    #     print(json.dumps(cur_results), file=log_f)

    # log
    # averaged_results = {
    #     'session': {},
    #     'turn': {}
    # }
    # for t in ['session', 'turn']:
    #     for k in results[0]['retrieval_results']['metrics'][t]:
    #         try:
    #             # will skip abstention instances for reporting the metric
    #             averaged_results[t][k] = np.mean([x['retrieval_results']['metrics'][t][k] for x in results if '_abs' not in x['question_id']])
    #         except:
    #             continue
    # print(json.dumps(averaged_results))

    # save results

    all_text = []

    # 遍历每个结果
    for i, entry in enumerate(results, start=1):
        # 遍历每个 ranked_items 列表中的元素
        for j, item in enumerate(entry["retrieval_results"]["ranked_items"], start=1):
            # 获取文本内容
            text_content = item["corpus_id"]
            # 按照期望的格式组合

            all_text.append(text_content)

    # 将所有的文本组合成一个完整的字符串
    # final_string = "\n\n".join(all_text)
    # for entry in results:
    # print(all_text)
    return all_text

    # log_f.close()
    # out_f.close()


if __name__ == "__main__":
    # args = parse_args()

    args = argparse.Namespace(
        in_file="/data/mufan/agent/LongMemEval/src/retrieval/output.jsonl",
        out_dir="/data/mufan/agent/LongMemEval",
        outfile_prefix=None,
        cache_dir=None,
        retriever="flat-contriever",
        granularity="session",
        index_expansion_method="none",
        index_expansion_llm="none",
        index_expansion_result_cache=None,
        index_expansion_result_join_mode="none",
        k=5,
        query="Summarized Instructions:\n**Summarizations of trajectory steps**  \nThe trajectory involved accessing a URL associated with the OpenStreetMap service. Upon navigating to the site, various services such as route finding and editing options were available. The actions included observing the map and the services provided, along with examining possible routes, directions, and edit functionalities.\n\n**END Summarizations of trajectory steps**\n\n**Abstractions of the trajectory purpose**  \nThe purpose of the trajectory was to explore and familiarize oneself with the OpenStreetMap interface, specifically to understand its maps, routing capabilities, and available services for user interaction and editing.\n\n**END Abstractions of the trajectory purpose**\n\n**yes**",
    )
    retrieval(args)

    # retrieval(in_file="/data/mufan/agent/LongMemEval/src/retrieval/output.jsonl",retriever="flat-contriever ", granularity="session",k=5,query="Summarized Instructions:\n**Summarizations of trajectory steps**  \nThe trajectory involved accessing a URL associated with the OpenStreetMap service. Upon navigating to the site, various services such as route finding and editing options were available. The actions included observing the map and the services provided, along with examining possible routes, directions, and edit functionalities.\n\n**END Summarizations of trajectory steps**\n\n**Abstractions of the trajectory purpose**  \nThe purpose of the trajectory was to explore and familiarize oneself with the OpenStreetMap interface, specifically to understand its maps, routing capabilities, and available services for user interaction and editing.\n\n**END Abstractions of the trajectory purpose**\n\n**yes**",home_dir="/data/mufan/agent/LongMemEval")
