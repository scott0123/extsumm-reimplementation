import json
import re
import rouge
from tqdm import tqdm
import sys

evaluator = rouge.Rouge(metrics=['rouge-n'],
                        max_n=1,
                        limit_length=False,
                        apply_avg=False,
                        apply_best=False,
                        alpha=0.5,  # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

"""
{
    "id": "",
    "inputs": [{
        "text": "",
        "tokens": [""],
        "sentence_id": (int), 
        "word_count": (int)
    }],
    "section_names": [""]
    "section_lengths": [(int)]
}
"""


class Processor:
    def __init__(self, original_data, output_dir="output_data", processing=0):
        self.original_data = original_data
        self.output = output_dir + "data_" + str(processing) + ".txt"
        self.label_output = output_dir + "label_" + str(processing) + ".txt"
        self.abstract_output = output_dir + "abstract_" + str(processing) + ".txt"
        self.processing = processing

    def process_sentence(self, sentence):
        return re.sub(r"\[([0-9, ]+)\]", "", sentence)

    def process_one_article(self, one_article):
        # ['article_id', 'article_text', 'abstract_text', 'labels', 'section_names', 'sections']
        output_dict = dict()
        output_dict['id'] = one_article['article_id']
        output_dict['inputs'] = []
        for sentence_id, sentence in enumerate(one_article['article_text']):
            sentence_dict = dict()
            sentence = self.process_sentence(sentence)
            sentence_dict['text'] = sentence
            sentence_dict['tokens'] = sentence.split()
            sentence_dict['sentence_id'] = sentence_id + 1
            sentence_dict['word_count'] = len(sentence_dict['tokens'])
            output_dict['inputs'].append(sentence_dict)
        output_dict['section_names'] = one_article['section_names']
        output_dict['section_lengths'] = [len(sec) for sec in one_article['sections']]
        return json.dumps(output_dict)

    def process(self):
        with open(self.original_data) as data_in:
            for i, article in enumerate(tqdm(data_in.read().splitlines())):
                if i % processing == 0:
                    article = json.loads(article)
                    abstract = self.process_abstract(' '.join(article['abstract_text']))
                    with open(self.abstract_output, "a+") as abstract_out:
                        abstract_out.write(abstract)
                        abstract_out.write('\n')
                    with open(self.output, "a+") as data_out:
                        data_out.write(self.process_one_article(article))
                        data_out.write('\n')
                    with open(self.label_output, "a+") as label_out:
                        label_dict = dict()
                        label_dict['id'] = article['article_id']
                        label_dict['labels'] = self.generate_label(article['article_text'], abstract, limit=200)
                        label_out.write(json.dumps(label_dict))
                        label_out.write('\n')

    @staticmethod
    def process_abstract(abstract):
        # return abstract
        abstract = re.sub("<[a-zA-Z/]*>", "", abstract)
        return ' '.join(abstract.split())

    @staticmethod
    def generate_label(article, reference, limit=200):
        hyp = ""
        wc = 0
        picked = []
        highest_r1 = 0
        sid = -1
        while wc <= limit:
            for i, sentence in enumerate(article):
                scores = evaluator.get_scores(hyp + sentence, reference)
                score = scores['rouge-1'][0]['f'][0]
                if score > highest_r1:
                    highest_r1 = score
                    sid = i
            if sid != -1:
                picked.append(sid)
                hyp += article[sid]
                wc += len(article[sid].split())
            else:
                break
        one_hot = [0 for _ in range(len(article))]
        for i in picked:
            one_hot[i] = 1
        return one_hot


if __name__ == '__main__':
    processing = int(sys.argv[1])
    print("running process ", processing)
    original_data = "./arxiv-dataset/test.txt"
    pre_process = Processor(original_data, output_dir="./preprocessed/test/", processing=processing)
    pre_process.process()


