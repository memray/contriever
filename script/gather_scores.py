import json
import os.path

'''
4 datasets are not directly available by BEIR: 'bioasq', 'signal1m', 'robust04', 'trec-news'
2 datasets are not in leaderboards by Mar 19, 2022: 'cqadupstack', 'quora'
'''
def main():
    exp_base_dir = '/export/home/exp/search/contriever/'
    exp_names = [
        # 'fb-contriever.dot',
        # 'fb-contriever.msmarco.dot',
        # 'sup-simcse-bert-base-uncased.dot',
        # 'unsup-simcse-bert-base-uncased.dot',
        # 'unsup-simcse-bert-large-uncased.dot',
        # 'unsup-simcse-roberta-base.dot',
        # 'sup-simcse-roberta-large.dot',
        'sup-simcse-roberta-large.cos_sim',
        # 'unsup-simcse-roberta-large.dot',
        # 'unsup-simcse-roberta-large.cos_sim',
        ]
    beir_datasets = [
        'msmarco',
        'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04',
        'quora', 'cqadupstack']
    # beir_datasets = [
    #     'msmarco',
    #     'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
    #     'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
    #     'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04'
    #     ]

    beir_metrics = ['ndcg', 'recall', 'map', 'mrr', 'precision', 'recall_cap', 'hole']
    core_metric = 'ndcg@10'
    exp2scores = {}

    for exp_name in exp_names:
        print('=-' * 20)
        print(exp_name.upper())
        print('=-' * 20)
        header_row = None
        score_rows = []

        for dataset in beir_datasets:
            if not header_row: _header_row = ['']
            score_row = [dataset]

            score_json_path = os.path.join(exp_base_dir, exp_name, f'{dataset}.json')
            if not os.path.exists(score_json_path):
                print(f'{dataset} not found at: {score_json_path}')
            else:
                print(dataset.upper())
                with open(score_json_path, 'r') as jfile:
                    result_data = json.load(jfile)
                # print(result_data)

                for metric_prefix in beir_metrics:
                    for metric, score in result_data['scores'][metric_prefix].items():
                        score_row.append(score)
                        if not header_row: _header_row.append(metric.lower())
                if not header_row: header_row = _header_row
            score_rows.append(score_row)

        exp2scores[exp_name] = score_rows

    for exp_name, score_rows in exp2scores.items():
        print('*' * 20)
        print(exp_name)
        print(','.join(header_row))
        for row in score_rows:
            print(','.join([str(e) for e in row]))


if __name__ == '__main__':
    main()
