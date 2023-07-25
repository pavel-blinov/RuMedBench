import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import precision_recall_curve

from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task


def lama_train(df_list, random_seed):

    roles = {
        "target": "targets",
        "category": "device"
    }

    # https://github.com/sb-ai-lab/LightAutoML
    # define that machine learning problem is binary classification
    task = Task("binary")

    utilized_automl = TabularUtilizedAutoML(
        task = task,
        timeout = 180,
        cpu_limit = 8,
        reader_params = {'n_jobs': 8, 'cv': 5, 'random_state': random_seed}
    )

    _ = utilized_automl.fit_predict(df_list[0], roles = roles, verbose = 1)

    # threshold search
    val_pred = utilized_automl.predict(df_list[1].drop(columns=["targets"]))
    precision, recall, thresholds = precision_recall_curve(df_list[1]["targets"], val_pred.data.squeeze())
    best_thrsh = thresholds[(2*recall*precision / (recall + precision)).argmax()]

    pub_pred = utilized_automl.predict(df_list[2])
    pub_res = pub_pred.data.squeeze() > best_thrsh
    return pub_res, utilized_automl
