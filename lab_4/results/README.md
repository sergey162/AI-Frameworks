# Сохранённые результаты лабораторной №4

Файлы формируются при полном выполнении [ноутбука](../random_forest_research.ipynb).
Главная таблица `metrics.csv` совместима со сводкой `comparison.ipynb`.

| Файлы | Содержание |
|:--|:--|
| `metrics.csv` | Все test-метрики, время полного Pipeline, идентификаторы данных и разбиений |
| `protocol.json`, `selected.json` | Полный протокол и выбранные по train-CV настройки |
| `split_*.csv`, `cv_folds.csv` | Исходные row_id, train/test, группы и назначения CV-фолдов |
| `predictions_*.csv` | Истина, test-прогнозы и вероятность класса 1 для каждой модели |
| `hypotheses.json`, `hypothesis_summary.csv` | Проверка представлений и изменения CV-score |
| `n_estimators_cv.csv` | Кривые train/CV и накопленное время обучения леса внутри фолда |
| `cv_classification.csv`, `cv_regression.csv` | Все совместные конфигурации и оценки каждого CV-фолда |
| `bootstrap_cv.json`, `seed_cv.json` | Контроль bootstrap и устойчивость к случайному состоянию |
| `complexity.csv`, `agreement.json` | Сложность, качество train и расхождения реализаций |
| `oob.csv`, `bootstrap_audit.csv` | OOB-метрики, покрытие, число голосов, уникальность bootstrap |
| `feature_importances.csv` | MDI улучшенного sklearn с группировкой one-hot |
| `permutation_importances.csv` | Перестановочная важность исходных признаков на CV-validation |
| `paired_bootstrap.csv` | Разности test-метрик и условные 95% percentile-интервалы |

CV-score — F1 для классификации и **отрицательная MAE** для регрессии: больше — лучше.
В `metrics.csv`, `complexity.csv` и `oob.csv` MAE записана в обычном положительном виде.
Основной OOB-показатель — F1 / MAE, стандартный `oob_score_` — accuracy / R².
OOB бетона исключает строки, но не целые рецептуры; параметры выбираются по GroupKFold.

В `feature_set` указаны исходные входы для проверки сопоставимости работ;
`representation` описывает их преобразование и добавление признаков.
Времена измерены одним запуском, включают предобработку, не включают CV-поиск.
Парные интервалы условны относительно фиксированных обученных моделей; регрессия
ресэмплирует целые test-рецептуры, классификация — строки.
