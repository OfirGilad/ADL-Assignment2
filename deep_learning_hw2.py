import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def task1(eps=1e-3, delta=1e-5, itr=10000):
    A = diabetes.data
    b = diabetes.target
    err_func = lambda x: np.linalg.norm(A @ x - b) ** 2
    err_graph = list()

    x0 = np.zeros(A.shape[1])
    x = x0
    for i in range(itr):
        grad = 2 * ((A.T @ A) @ x - A.T @ b)
        x = x - eps * grad
        err = err_func(x)
        err_graph.append(err)
        if np.linalg.norm(grad) < delta:
            print(f'GD converged after {i} iterations')
            break

    err_graph = np.stack(err_graph)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Task 1: Gradient Descent Errors')
    plt.plot(err_graph, label='Error')
    plt.legend()
    plt.show()


def task2(eps=1e-3, delta=1e-5, itr=1000):
    A_train, A_test, b_train, b_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)
    err_func_train = lambda x: np.linalg.norm(A_train @ x - b_train) ** 2
    err_func_test = lambda x: np.linalg.norm(A_test @ x - b_test) ** 2
    err_graph_train = list()
    err_graph_test = list()

    x0 = np.zeros(A_train.shape[1])
    x = x0
    for i in range(itr):
        grad = 2 * ((A_train.T @ A_train) @ x - A_train.T @ b_train)
        x = x - eps * grad
        train_err = err_func_train(x)
        test_err = err_func_test(x)
        err_graph_train.append(train_err)
        err_graph_test.append(test_err)
        if np.linalg.norm(grad) < delta:
            print(f'GD converged after {i} iterations')
            break

    err_graph_train = np.stack(err_graph_train)
    err_graph_test = np.stack(err_graph_test)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Task 2: Gradient Descent Train and Test Errors')
    plt.plot(err_graph_train, label='Train Error')
    plt.plot(err_graph_test, label='Test Error')
    plt.legend()
    plt.show()


def task3(eps=1e-3, delta=1e-5, itr=1000):
    all_err_train = list()
    all_err_test = list()
    for rounds in range(10):
        A_train, A_test, b_train, b_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
        err_func_train = lambda x:  np.linalg.norm(A_train @ x - b_train) ** 2
        err_func_test = lambda x:  np.linalg.norm(A_test @ x - b_test) ** 2
        err_graph_train = list()
        err_graph_test = list()

        x0 = np.random.rand(A_train.shape[1])
        x = x0
        for i in range(itr):
            grad = 2 * ((A_train.T @ A_train) @ x - A_train.T @ b_train)
            x = x - eps * grad
            err_train = err_func_train(x)
            err_test = err_func_test(x)
            err_graph_train.append(err_train)
            err_graph_test.append(err_test)
            if np.linalg.norm(grad) < delta:
                print(f'GD converged after {i} iterations')
                break

        all_err_train.append(err_graph_train)
        all_err_test.append(err_graph_test)

    # Compute average and minimum errors
    avg_train_err = np.mean(np.array(all_err_train), axis=0)
    avg_test_err = np.mean(np.array(all_err_test), axis=0)
    min_train_err = np.min(np.array(all_err_train), axis=0)
    min_test_err = np.min(np.array(all_err_test), axis=0)

    # Avg
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_err, label='Average Train Error')
    plt.plot(avg_test_err, label='Average Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Task 3: Gradient Descent Average of Train and Test Errors')
    plt.legend()
    plt.show()

    # Min
    plt.figure(figsize=(10, 5))
    plt.plot(min_train_err, label='Minimum Train Error')
    plt.plot(min_test_err, label='Minimum Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Task 3: Gradient Descent Minimum of Train and Test Errors')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    diabetes = load_diabetes()

    task1()
    task2()
    task3()
