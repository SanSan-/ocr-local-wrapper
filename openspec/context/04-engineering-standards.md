# Стандарты и команды

Обязательны [общий профиль Python](../specs/engineering-governance/reference/python-common.md)
и [локальные правила](../specs/engineering-governance/reference/project-specific.md). Там указаны дополнительные
ограничения и команды.

Пакет ocr_local версии 0.1.0; Python >=3.14. Прямой и транзитивный набор закреплён в requirements.txt; pyproject.toml
содержит metadata/pytest и dependencies=[]. Это не uv-проект. Ruff, отдельный Sonar-контур и lock-файл в текущем конфиге
не объявлены.

Для документации из корня: `openspec validate --all --strict --no-interactive`, ссылки и UTF-8 без BOM. Основной pytest
и тяжёлый GLM-тест имеют отдельные границы; их описание не является разрешением выполнить OCR при работе с
документацией.
