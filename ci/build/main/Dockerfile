# Используем официальный образ PyTorch с более новой версией CUDA 11.7 и PyTorch 2.0
ARG BASE_IMG
FROM ${BASE_IMG}

# Устанавливаем рабочую директорию внутри контейнера
ARG WORKDIR
WORKDIR ${WORKDIR}

# Копируем файл зависимостей
COPY requirements.txt ${WORKDIR}/requirements.txt

# Устанавливаем зависимости
RUN pip install --upgrade pip && pip install -r ${WORKDIR}/requirements.txt


