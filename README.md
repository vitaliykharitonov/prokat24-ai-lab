# Prokat24 AI Lab

[![Check](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml/badge.svg)](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml)

Учебный AI/Evals проект на базе сайта prokat24.com.ua.

## Цели

- Изучить Python через реальные задачи.
- Построить базу знаний по товарам и правилам проката.
- Сделать AI-чат для клиентов.
- Построить evals-систему для проверки качества AI-ответов.

## Настройка окружения для разработки

Создать виртуальное окружение:

```
python -m venv .venv
```

Активировать виртуальное окружение:

```
.\.venv\Scripts\Activate.ps1
```

Установить проект и dev-зависимости:

```
pip install -e ".[dev]"
```

Запустить полную проверку проекта:

```
python src/check_project.py
```

## Быстрая проверка проекта

Главная команда перед коммитом:

[Открыть скрипт проверки проекта](src/check_project.py)

```
python src/check_project.py
```

Проверки:

- pytest
- ruff check
- ruff format --check
- search evals
- answer evals
- product answer evals

Полные JSON eval reports можно создать отдельными командами из раздела `Evals`

Ожидаемый результат: все проверки завершаются успешно

## Основные команды

Запуск всех тестов:

```
pytest
```

Проверка качества кода:

```
ruff check .
```

Проверка форматирования:

```
ruff format --check .
```

Автоматическое форматирование:

```
ruff format .
```

Проверка полного проекта:

```
python src/check_project.py
```

## Поиск по продуктам CLI

Примеры пользовательских запросов:

```
python scripts/cli/search_products.py "скільки коштує 2 місний намет на тиждень"
python scripts/cli/search_products.py "яка застава за 2 місний намет"
python scripts/cli/search_products.py "прокат спальника на 2 тижні"
```

(Сейчас товарные запросы в answer layer ориентированы на украинский язык)

## Каталог продуктов

Экспорт каталога товаров:

```
python scripts/data/export_product_catalog.py
```

Команда обновляет файл (на основе страниц сайта):

```
data/products.json
```

## Evals

В проекте есть три eval-направления:

- `product_answer_eval_report` — проверки ответов на товарные запросы.
- `search_eval_report` — проверки качества поиска по базе знаний.
- `answer_eval_report` — проверки ответов по базе знаний.

Запуск:

```powershell
python scripts/evals/run_product_answer_evals.py
python scripts/evals/run_search_evals.py
python scripts/evals/run_answer_evals.py
```

JSON-отчёты сохраняются в `reports/` и не коммитятся в Git.

Все eval reports содержат metadata:

```text
report_type
schema_version
created_at
git_commit
git_dirty
```

`git_commit` и `git_dirty` помогают понять, на какой версии кода был создан отчёт и были ли незакоммиченные изменения.

## Рабочий цикл разработки

Обычный порядок работы:

1. Изменить код.
2. Добавить или обновить тесты.
3. Запустить проверку проекта:

```
python src/check_project.py
```

4. Если всё зелёное — сделать commit.
5. Отправить изменения на GitHub:

```powershell
git push
```

6. Проверить, что GitHub Actions workflow `Check` прошёл успешно.

Для изменений в отдельной ветке используется pull request:

1. Создать feature branch.
2. Сделать изменения и commit.
3. Отправить ветку на GitHub.
4. Открыть pull request в `main`.
5. Дождаться зелёного GitHub Actions check.
6. После проверки выполнить merge.