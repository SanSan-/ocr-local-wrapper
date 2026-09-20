# Архитектура

1. CLI и веб разрешают OcrOptions через общий реестр профилей GLM/Paddle.
2. Проверки изображения и профильного кеша предшествуют загрузке весов.
3. При промахе используются только локальные processor/model. Веб получает модель из pool ёмкости 1; при смене
   освобождает прежнюю.
4. CUDA использует прямое размещение, BF16/FP16 и SDPA, INT8 только явно. Генерация: inference_mode, use_cache, greedy.
   CPU fallback зависит от параметра.
5. Общий сервис очищает ответ, отклоняет пустой, пишет TXT/кеш и измеряет стадии.
6. Веб копирует вход в uploads, хранит отдельные TXT по настройкам и выполняет один поток; ошибка старта освобождает
   слот.

Компоненты: [model_profiles.py](../../ocr_local/model_profiles.py), [service.py](../../ocr_local/service.py), [model_utils.py](../../ocr_local/utils/model_utils.py), [output_cache.py](../../ocr_local/utils/output_cache.py), [web/app.py](../../ocr_local/web/app.py).
Причины решений — [ADR](ADR/README.md).
