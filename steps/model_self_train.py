import logging
import logging.config
import warnings
import joblib
from steps_context import tag, label_folder, model_folder, compare_folder, logger_conf
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.models.model_build import semi_training

warnings.filterwarnings("ignore")


def main(form_label, y_col, model_name, threshold=0.95):

    logging.config.fileConfig(logger_conf)
    logger = logging.getLogger('model_self_train')

    logger.info(f'Self training for {form_label} {y_col} using {model_name}')


    # * Preparing model
    model = joblib.load(model_folder / f'{form_label}_{y_col}_{model_name}.joblib')


    # * Preparing files
    sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx'
    sents_labeled_1 = label_folder / f'{tag}_{form_label}_sents_labeled_1.xlsx'
    labeled = merge_csv([sents_labeled, sents_labeled_1])
    labeled.drop('cik', axis=1, inplace=True)
    labeled.fillna(0, inplace=True)
    labeled.drop_duplicates(inplace=True)

    sents_unlabeled = label_folder / f'{tag}_{form_label}_sents_unlabel.xlsx'
    unlabeled = read_file_df(sents_unlabeled)

    data = semi_training(labeled, unlabeled, y_col=y_col, x_col='sents')
    data.gen_train_test_set()
    data.prepare_unlabeled_set()

    # * self training
    data.self_training(model, threshold=threshold)

    # * self training results
    new_labeled = data.self_training_result()
    to_file_df(new_labeled, label_folder / f'{form_label}_{y_col}_label_propagated.xlsx')
    logger.info('self training results saved')

    # * self training check
    label_chg = data.self_training_check()
    to_file_df(label_chg, compare_folder / f'{form_label}_{y_col}_{model_name}_label_chg.xlsx')
    logger.info('self training changed label saved')

    # * self training no result
    label_noresult = data.self_training_noresult()
    to_file_df(label_noresult, compare_folder / f'{form_label}_{y_col}_{model_name}_label_nr.xlsx')
    logger.info('self training no result saved')

    # * new model performance
    data.find_best_threshold()
    data.model_predict()
    data.model_val()
    data.model_sum['model_name'] = model_name
    logger.info('model performance after self training')
    logger.info(f'{data.model_sum}')


if __name__ == "__main__":
    # main('CR', 'Incident', 'Baseline')
    # main('PR', 'Incident', 'Baseline', threshold=0.99)
    main('PR', 'Immaterial', 'Baseline', threshold=0.99)