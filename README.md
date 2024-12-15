# Tera Link Hackaton
# IDR Rep solution

# Шаблон

- В этой ветке лежит шаблон всего проекта - по сути, структура
 
    Его предполагается никак не изменять(только дополнять, например, тестами)

- Все остальные ветки должны содержать данную структуру:

    1. Совсем новые по функциональности ветки создаются от данной 
    2. Если разработка велась ранее, необходимо замержить в раннюю ветку данную,
    при этом ранние файлы поместить в соответсвующее место

- Разработка данной структуры велась с помощью пакетного и проектного менеджера `uv`,
файлы в корне проекта: `.python-version`, `pyproject.toml` и `uv.lock`- созданы им
  - `.python-version` - хранит в себе используемую версию питона - 3.10
  - `pyproject.toml` - описание проекта и зависимостей в человеко-читаемом формате
  - `uv.lock` - системный файл для хранения установленных зависимостей, 
  (почти) автоматически генерируется `uv`

- `requirements.txt` - файл с зависимостями стандартного вида, для общих нужд

- В дальнейшем для запуска приложения предполагается что-то вроде:

  ```shell
  python -m app
  ```

## team:
- Ежов Дмитрий Александрович
- Трифонов Василий Максимович
- Соловьев Матвей Михайлович
- Вдовин Герман Евгеньевич

# case:
#### Нейросеть по распознаванию текста (OCR)

Классификация документов, автоматическое извлечение данных, проверка орфографии, пунктуации, соответствии СТО компании.
В цифровом мире широко развивается и применяется электронный документооборот. Для упрощения и автоматизации разбора входящей корреспонденции системы электронного документооборота необходимо применить технологии искусственного интеллекта для автоматизации обработки документов и повышения эффективности работы компании.

[Техническое описание задачи](https://contestfiles.storage.yandexcloud.net/companies/codenrock-13/contests/terralink-codefest/%D0%A2%D1%80%D0%B5%D0%BA_3_%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D1%8C_%D0%BF%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D1%8E_%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%B0_%28OCR%29%2C_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8_%D0%B4%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D0%BE%D0%B2_%D0%B8_%D0%B0%D0%B2%D1%82%D0%BE%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%BC%D1%83_%D0%B8%D0%B7%D0%B2%D0%BB%D0%B5%D1%87%D0%B5%D0%BD%D0%B8%D1%8E_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85.pdf)