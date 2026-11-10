# Источники и описание данных

Для классификации и регрессии используются **разные наборы**, проверенные по таблице группы. Результат проверки уникальности приведён в [uniqueness_check.json](uniqueness_check.json).

## Классификация: Loan Approval Classification Data

Автор **taweilo**; [первичный источник Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data).

- Файл `raw/loan_data.csv`: 45 000 строк, 14 столбцов, сохранён без изменений.
- Kaggle download API при подготовке вернул HTTP 404. Файл получен из [публичной копии с фиксированным commit](https://github.com/dpidaparthi/loan-approval-classification/blob/379a24f61cd5d9f032415b14a5797f30cf1d4a05/data/loan_data.csv).
- SHA256: `67cca604cd71cbd605014f721d078d20d44d726d625945591450733535f9868a`.
- Прямое побайтовое сравнение с недоступной загрузкой Kaggle не выполнено; SHA256 относится к включённой копии.
- Условия использования определяются исходным набором Kaggle; новая лицензия на данные не назначается.

Описание набора указывает синтетическое происхождение и расширение SMOTENC. Возможная зависимость синтетических объектов ограничивает интерпретацию holdout. Цель — исторический статус заявки; она не эквивалентна будущему дефолту.

| Столбец | Смысл |
|---|---|
| `person_age` | Возраст |
| `person_gender` | Пол |
| `person_education` | Образование |
| `person_income` | Годовой доход |
| `person_emp_exp` | Стаж |
| `person_home_ownership` | Тип владения жильём |
| `loan_amnt` | Запрашиваемая сумма кредита, вход классификации |
| `loan_intent` | Цель кредита |
| `loan_int_rate` | Ставка; исключена для сценария до её назначения |
| `loan_percent_income` | Отношение запрашиваемой суммы к доходу |
| `cb_person_cred_hist_length` | Длительность кредитной истории |
| `credit_score` | Кредитный рейтинг |
| `previous_loan_defaults_on_file` | Наличие прошлых дефолтов |
| `loan_status` | Цель: 1 — одобрено, 0 — отказ |

Для классификации используется 12 входных столбцов; `loan_amnt` входит в состав признаков.

## Регрессия: Concrete Compressive Strength

**Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning Repository.**
[Страница UCI №165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), [DOI: 10.24432/C5PK67](https://doi.org/10.24432/C5PK67).

- Файл `raw/Concrete_Data.xls`: 1030 измерений, восемь входных признаков и одна цель.
- Получен из [официального архива UCI](https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip); исходный XLS сохранён без изменений.
- SHA256 XLS: `710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4`.
- Лицензия UCI: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- При чтении изменяются только имена столбцов; значения и исходные номера строк сохраняются.

| Короткое имя | Исходная величина | Единицы |
|---|---|---|
| `cement` | Цемент | кг/м³ |
| `slag` | Доменный шлак | кг/м³ |
| `fly_ash` | Зола-унос | кг/м³ |
| `water` | Вода | кг/м³ |
| `superplasticizer` | Суперпластификатор | кг/м³ |
| `coarse_aggregate` | Крупный заполнитель | кг/м³ |
| `fine_aggregate` | Мелкий заполнитель | кг/м³ |
| `age` | Возраст образца | дни |
| `strength` | Цель: прочность при сжатии | МПа |

У исходного набора нет пропусков. В исследовании удаляются 25 точных дубликатов полных строк, остаётся 1005 наблюдений. Измерения одной рецептуры (одинаковые семь компонентов, независимо от возраста) объединяются в группу. Разделение GroupShuffleSplit и GroupKFold исключает пересечение групп. Это предотвращает узнавание смеси, уже встречавшейся в другом возрасте; целью остаётся прогноз прочности для новых рецептур.

## Уникальность и воспроизводимость

Выбор проверен по [таблице группы](https://docs.google.com/spreadsheets/d/1szg1vYr3MLq8eqCejGbvVOdF7t87slb3h2x8qv2e-So/edit?gid=0) 10 ноября 2026 года. Loan Approval Classification Data и Concrete Compressive Strength не совпадают с датасетами других участников.

Оба исходных файла включены в репозиторий. Скрипт `download_data.py` проверяет SHA256 и при необходимости восстанавливает данные из фиксированной копии Kaggle и официального архива UCI. Цели, признаки и разбиения зафиксированы в `results/protocol.json`.
