##  Модель машинного обучения для соревнования kaggle игры ConnectX

- Мы используем возможности параллелизма [PARL]  для выполнения самостоятельных игр между только что обученной 
  моделью и лучшей версией предыдущих версий обучения.
- Мы также предоставляем сценарии для упаковки вашей хорошо обученной модели в файл для отправки в Kaggle [Connect X]
  (https://www.kaggle.com/c/connectx/leaderboard) напрямую.

### Необходимые пакеты для работы 
- python3
- parl
- torch
- tqdm

### Для начала работы


```commandline

# для создания списка установленных пакетов используйте команду
pip freeze > requirements.txt


# для установки всего списка необходимых пакетов
pip3 install -r requirements.txt
```


### Тренировка модели
1. Для первоначального обучения мы используем 1000 игр между двумя произвольными агентами с оценкой результатов игры.
   Вы также можете скачать этот датасет самостоятельно  по следующей ссылке - [1k connect4 validation 
   set](https://www.kaggle.com/petercnudde/1k-connect4-validation-set) 

2. Для начала работу необходимо запустить кластер xparl следующей командой в терминале:
```bash
# Вы можете изменить следующие `cpu_num` и` args.actor_nums` в main.py
# в зависимости от количества ядер Центрального Процессора вашего компьютера.
# у меня на ноутбуке их 8, для отображения прогресса тренировки запускаем веб-интерфейс, он в режиме реального 
# времени покажет нагрузку на процессор, а также отобразит работу каждого агента (ядра Вашего ЦП)

xparl start --port 8010 --cpu_num 8

#xparl start --port 8000 --debug --cpu_num 8 --monitor_port 8001 --log_server_port_range 8002-8010

# для получения справочной информации введите
xparl --help

# или
xparl connect --help
#xparl connect --address 127.0.0.1 --cpu_num 8 --log_server_port_range 8011-8019
```

3. Далее в терминале запустите тренировочный скрипт main.py:
```bash
# это программа для запуска полного цикла тренировки Вашей модели (использует метод обучения с подкреплением)
python /home/user/PycharmProjects/Kaggle_ConnectX/main.py
```

4. Для проверки статуса работы кластера xparl введите в терминале:
```commandline
xparl status
```

5. Для завершения работы кластера xparl введите в терминале:
```commandline
xparl stop
```

### Отправка результата Вашей работы
Чтобы отправить хорошо обученную модель в Kaggle, вы можете использовать предоставленный скрипт для создания 
файла submission.py, для этого введите в терминале:
```bash
python /home/user/PycharmProjects/Kaggle_ConnectX/gen_submission.py /home/user/PycharmProjects/Kaggle_ConnectX/saved_model/best.pth.tar
```

### запуск игры с обученной моделью и Вами
```commandline
python /home/user/PycharmProjects/Kaggle_ConnectX/connect4_aiplayer.py
```

### Полезные ссылки
Вводные блокноты
- [ConnectX Getting Started](https://www.kaggle.com/andrej0marinchenko/connectx-getting-started)
- [СonnectX second step](https://www.kaggle.com/andrej0marinchenko/onnectx-second-step)
