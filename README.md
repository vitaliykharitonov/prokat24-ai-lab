# Prokat24 AI Lab

[![Check](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml/badge.svg)](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml)

Учебный AI/Evals проект на базе сайта prokat24.com.ua.

## Цели

- Изучить Python через реальные задачи.
- Построить базу знаний по товарам и правилам проката.
- Сделать AI-чат для клиентов.
- Построить evals-систему для проверки качества AI-ответов.

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
- product answer evals

Ожидаемый результат:

```
Product answer evals: 23/23 passed
```

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
python src/search_products.py "скільки коштує 2 місний намет на тиждень"
python src/search_products.py "яка застава за 2 місний намет"
python src/search_products.py "прокат спальника на 2 тижні"
```

(Сейчас товарные запросы в answer layer ориентированы на украинский язык)

## Каталог продуктов

Экспорт каталога товаров:

```
python src/export_product_catalog.py
```

Команда обновляет файл (на основе страниц сайта):

```
data/products.json
```

## Проверка ответов

Запуск product answer evals:

```
python src/run_product_answer_evals.py
```

(Проверяют пользовательские запросы к товарному answer engine)

Пример успешного результата:

```
Product answer evals: 5/5 passed
```

Если eval падает, runner показывает:

```
Query
Missing fragments
Answer
```

и завершает процесс с ненулевым exit code.

## Рабочий цикл разработки

Обычный порядок работы:

1. Изменить код.
2. Добавить или обновить тесты.
3. Запустить проверку проекта:

```
python src/check_project.py
```

4. Если всё зелёное — сделать commit.