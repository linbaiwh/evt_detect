import logging
import logging.config
import warnings
import joblib
from steps_context import tag, label_folder, model_folder, compare_folder, input_folder, logger_conf
from evt_detect.utils.file_io import read_file_df, to_file_df, merge_csv
from evt_detect.models.model_build import semi_training
from evt_detect.utils.preprocess import rm_features, select_breach_firms


warnings.filterwarnings("ignore")


def main(form_label, y_col, model_name, threshold=0.95, save_model=False, ciks=False):

    logging.config.fileConfig(logger_conf)
    logger = logging.getLogger('model_self_train')

    logger.info(f'Self training for {form_label} {y_col} using {model_name}')


    # * Preparing model
    model = joblib.load(model_folder / f'{form_label}_{y_col}_{model_name}.joblib')


    # * Preparing files
    sents_labeled = label_folder / f'{tag}_{form_label}_sents_labeled.xlsx'

    # sents_labeled = label_folder / f'{form_label}_{y_col}_labeled.xlsx'
    labeled = read_file_df(sents_labeled)

    labeled.fillna(0, inplace=True)
    labeled.drop_duplicates(inplace=True)

    sents_unlabeled = label_folder / f'{tag}_{form_label}_sents_unlabel.xlsx'
    unlabeled = read_file_df(sents_unlabeled)
    if ciks:
        breached_ciks = select_breach_firms(input_folder)
        unlabeled = unlabeled.loc[unlabeled['cik'].isin(breached_ciks)]

    rm_cols = ['cik', 'Related', 'Incident', 'Immaterial', 'Cost', 'Litigation', 'Management']

    features = rm_features(unlabeled, rm_cols)

    data = semi_training(labeled, unlabeled, y_col=y_col, X_col=features)
    data.gen_train_test_set()
    data.prepare_unlabeled_set()

    # * self training
    data.self_training(model, threshold=threshold)

    # * self training results
    data.self_training_result()
    new_labeled = data.self_training_labeled()
    to_file_df(new_labeled, label_folder / f'{form_label}_{y_col}_label_propagated.xlsx')
    logger.info('self training results saved')

    # * self training check
    label_chg = data.self_training_chg()
    to_file_df(label_chg, compare_folder / f'{form_label}_{y_col}_{model_name}_label_chg.xlsx')
    logger.info('self training changed label saved')

    # * self training no result
    nolabel = data.self_training_nolabel()
    to_file_df(nolabel, compare_folder / f'{form_label}_{y_col}_{model_name}_nolabel.xlsx')
    logger.info('self training no result saved')

    # * new model performance
    # data.find_best_threshold(use_test=True)
    # data.train_test_predict()
    # data.model_scores()
    # data.model_sum['model_name'] = model_name
    # logger.info('model performance after self training')
    # logger.info(f'{data.model_sum}')

    # * save model
    if save_model:
        data.model_save(model_folder / f'{form_label}_{y_col}_{model_name}_self_train.joblib')


if __name__ == "__main__":
    # main('CR', 'Incident', 'Baseline', threshold=0.99)
    # main('CR', 'Related', 'Baseline', threshold=0.99)
    # main('PR', 'Incident', 'Baseline', threshold=0.99)
    main('PR', 'Related', 'Baseline_Robust', threshold=0.99)
    # main('PR', 'Immaterial', 'Baseline', threshold=0.99)