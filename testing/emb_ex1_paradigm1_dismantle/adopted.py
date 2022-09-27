#


#
from octis.dataset.dataset import Dataset


#
from topicx.baselines.cetopictm import CETopicTM
from topicx.utils import prepare_dataset


#
if __name__ == '__main__':

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder("../dataset")

    tm = CETopicTM(dataset=dataset,
                   topic_model='cetopic',
                   num_topics=5,
                   dim_size=5,
                   word_select_method='tfidf_idfi',
                   embedding='princeton-nlp/unsup-simcse-bert-base-uncased',
                   seed=42)

    tm.train()
    td_score, cv_score, npmi_score = tm.evaluate()
    print(f'td: {td_score} npmi: {npmi_score} cv: {cv_score}')

    topics = tm.get_topics()
    print(f'Topics: {topics}')

