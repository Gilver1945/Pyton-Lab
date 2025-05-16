y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

#доля правильных предсказаний
def accuracy(y_true, y_pred):
    if len(y_true) == len(y_pred):
        correct = 0
        for idx in range(len(y_true)):
            if y_pred[idx] == y_true[idx]:
                correct += 1
        return correct / len(y_true)
    else:
        print("Ошибка")


print("Точность:", accuracy(y_true, y_pred))

#Возвращает precision (прецизионность) и recall (полноту).
def precision_recall(y_true, y_pred):
    if len(y_true) == len(y_pred):
        true_positive = 0
        false_negative = 0
        false_positive = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                true_positive += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                false_negative += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                false_positive += 1

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        return "Precision:", precision, "Recall:", recall
    else:
        print("Ошибка")


print(precision_recall(y_true, y_pred))

#средняя квадратичная ошибка
def mean_squared_error(y_true, y_pred):
    if len(y_true) == len(y_pred):
        total_error = 0
        for i in range(len(y_true)):
                total_error += (y_pred[i] - y_true[i]) ** 2
        return total_error / len(y_true)
    else:
        print("Ошибка")


print("MSE:", mean_squared_error(y_true, y_pred))

#средняя абсолютная ошибка
def mean_absolute_error(y_true, y_pred):
    if len(y_true) == len(y_pred):
        total_error = 0
        for i in range(len(y_true)):
            total_error += abs(y_pred[i] - y_true[i])
        return total_error / len(y_true)
    else:
        print("Ошибка")


print("MAE:", mean_absolute_error(y_true, y_pred))