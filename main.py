import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def task1(eps=1e-3, delta=1e-5, itr=10000):
    A = diabetes.data
    b = diabetes.target
    err_func = lambda x: .5 * np.linalg.norm(A @ x - b) ** 2
    err_graph = list()

    x0 = np.zeros(A.shape[1])
    x = x0
    for i in range(itr):
        grad = (A.T @ A) @ x - A.T @ b
        x = x - eps * grad
        err = err_func(x)
        err_graph.append(err)
        if np.linalg.norm(grad) < delta:
            print(f'GD converged after {i} iterations')
            break

    err_graph = np.stack(err_graph)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Error Plot of Gradient Descent')
    plt.plot(err_graph)
    plt.show()


def task1GPT():
    A = diabetes.data
    b = diabetes.target

    # Gradient Descent parameters
    eps = 0.001  # Step size (learning rate)
    delta = 1e-6  # Stopping condition
    max_iterations = 2000  # Maximum number of iterations

    # Initialize variables
    n_samples, n_features = A.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    error_plot = []  # To store error values

    # Define the error function
    def compute_error(A, x, b):
        return np.linalg.norm(A @ x - b) ** 2

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A.T @ A @ x - A.T @ b)  # Gradient of the least squares cost function
        x = x - eps * grad  # Update the parameters
        error = compute_error(A, x, b)  # Compute the error
        error_plot.append(error)  # Store the error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            print(f'Gradient Descent converged after {i} iterations')
            break

    # If gradient descent did not converge within the maximum number of iterations
    if i == max_iterations - 1:
        print('Gradient Descent did not converge within the maximum number of iterations')

    # Final optimized x
    print('Optimized x:', x)

    # Plotting the error over iterations
    plt.plot(error_plot)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Error Plot of Gradient Descent')
    plt.show()


def task2GPT(max_iterations=100000, eps=0.001, delta=1e-6):
    # Load diabetes dataset
    diabetes = load_diabetes()
    A = diabetes.data  # Feature matrix (442 x 10)
    b = diabetes.target  # Target vector (442,)

    # Split the dataset into training and test sets
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)

    # Gradient Descent parameters
    # eps = 0.001  # Step size (learning rate)
    # delta = 1e-6  # Stopping condition
    # max_iterations = 1000  # Maximum number of iterations

    # Initialize variables
    n_samples, n_features = A_train.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    train_error_plot = []  # To store training error values
    test_error_plot = []  # To store test error values

    # Define the error function
    def compute_error(A, x, b):
        return np.linalg.norm(A @ x - b) ** 2

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A_train.T @ A_train @ x - A_train.T @ b_train)  # Gradient of the least squares cost function
        x = x - eps * grad  # Update the parameters

        train_error = compute_error(A_train, x, b_train)  # Compute the training error
        test_error = compute_error(A_test, x, b_test)  # Compute the test error

        train_error_plot.append(train_error)  # Store the training error
        test_error_plot.append(test_error)  # Store the test error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            print(f'Gradient Descent converged after {i} iterations')
            break

    # If gradient descent did not converge within the maximum number of iterations
    if i == max_iterations - 1:
        print('Gradient Descent did not converge within the maximum number of iterations')

    # Final optimized x
    print('Optimized x:', x)

    # Plotting the train and test errors over iterations
    plt.plot(train_error_plot, label='Train Error')
    plt.plot(test_error_plot, label='Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Train and Test Error Plot of Gradient Descent')
    plt.legend()
    plt.show()


def task2(eps=1e-3, delta=1e-5, itr=10000):
    A_train, A_test, b_train, b_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)
    err_fun_train = lambda x: .5 * np.linalg.norm(A_train @ x - b_train) ** 2
    err_fun_test = lambda x: .5 * np.linalg.norm(A_test @ x - b_test) ** 2
    err_graph_train = list()
    err_graph_test = list()

    x0 = np.zeros(A_train.shape[1])
    x = x0
    for i in range(itr):
        grad = (A_train.T @ A_train) @ x - A_train.T @ b_train
        x = x - eps * grad
        train_err = err_fun_train(x)
        test_err = err_fun_test(x)
        err_graph_train.append(train_err)
        err_graph_test.append(test_err)
        if np.linalg.norm(grad) < delta:
            print(f'GD converged after {i} iterations')
            break

    err_graph_train = np.stack(err_graph_train)
    err_graph_test = np.stack(err_graph_test)
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Train and Test Error Plot of Gradient Descent')
    plt.plot(err_graph_train, label='Train Error')
    plt.plot(err_graph_test, label='Test Error')
    plt.legend()
    plt.show()


def compute_error(A, x, b, alpha=0):
    return np.linalg.norm(A @ x - b) ** 2 + alpha * np.linalg.norm(x) ** 2


def gradient_descent_ridge(A_train, b_train, A_test, b_test, eps=0.001, delta=1e-6, max_iterations=1000, alpha=1.0):
    n_samples, n_features = A_train.shape
    x0 = np.zeros(n_features)  # Initial guess (zero vector)
    train_error_plot = []  # To store training error values
    test_error_plot = []  # To store test error values

    # Gradient Descent Loop
    x = x0
    for i in range(max_iterations):
        grad = 2 * (A_train.T @ (A_train @ x - b_train) + alpha * x)  # Gradient with regularization
        x = x - eps * grad  # Update the parameters

        train_error = compute_error(A_train, x, b_train, alpha)  # Compute the training error
        test_error = compute_error(A_test, x, b_test, alpha)  # Compute the test error

        train_error_plot.append(train_error)  # Store the training error
        test_error_plot.append(test_error)  # Store the test error

        if np.linalg.norm(grad) < delta:  # Stopping condition
            break

    return train_error_plot, test_error_plot


def task3GPT():
    # Load diabetes dataset
    A = diabetes.data  # Feature matrix (442 x 10)
    b = diabetes.target  # Target vector (442,)

    num_runs = 10
    max_iterations = 1000
    eps = 0.001
    delta = 1e-6
    alpha = 1.0

    train_errors = np.zeros((num_runs, max_iterations))
    test_errors = np.zeros((num_runs, max_iterations))

    for run in range(num_runs):
        A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=run)

        train_error_plot, test_error_plot = gradient_descent_ridge(A_train, b_train, A_test, b_test, eps, delta,
                                                                   max_iterations, alpha)

        # Pad with the last error if the iteration count is less than max_iterations
        train_errors[run, :len(train_error_plot)] = train_error_plot
        train_errors[run, len(train_error_plot):] = train_error_plot[-1]
        test_errors[run, :len(test_error_plot)] = test_error_plot
        test_errors[run, len(test_error_plot):] = test_error_plot[-1]

    # Compute average and minimum errors
    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)
    min_train_errors = np.min(train_errors, axis=0)
    min_test_errors = np.min(test_errors, axis=0)

    # Plotting average errors
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_errors, label='Average Train Error')
    plt.plot(avg_test_errors, label='Average Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Average Train and Test Error Plot')
    plt.legend()
    plt.show()

    # Plotting minimum errors
    plt.figure(figsize=(10, 5))
    plt.plot(min_train_errors, label='Minimum Train Error')
    plt.plot(min_test_errors, label='Minimum Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Minimum Train and Test Error Plot')
    plt.legend()
    plt.show()


def average_and_minimum_graphs(graphs):
    """
    Receives an array of arrays, where each array is data for a graph, and returns
    one array that represents the average of these graphs and another array that
    represents the minimum of these graphs.

    Parameters:
    graphs (list of list of float): List of graphs, where each graph is represented as a list of float values.

    Returns:
    tuple: A tuple containing two arrays:
           - average_graph (np.array): An array representing the average of the input graphs.
           - minimum_graph (np.array): An array representing the minimum of the input graphs.
    """
    # Convert the list of lists to a numpy array for easier manipulation
    graphs_np = np.array(graphs)

    # Calculate the average across the graphs (column-wise)
    average_graph = np.mean(graphs_np, axis=0)

    # Calculate the minimum across the graphs (column-wise)
    minimum_graph = np.min(graphs_np, axis=0)

    return average_graph, minimum_graph


def task3(eps=1e-3, delta=1e-5, itr=10000):
    acc_train_err = list()
    acc_test_err = list()
    for rounds in range(10):
        A_train, A_test, b_train, b_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
        err_fun_train = lambda x: .5 * np.linalg.norm(A_train @ x - b_train) ** 2
        err_fun_test = lambda x: .5 * np.linalg.norm(A_test @ x - b_test) ** 2
        err_graph_train = list()
        err_graph_test = list()

        x0 = np.random.rand(A_train.shape[1])
        x = x0
        for i in range(itr):
            grad = (A_train.T @ A_train) @ x - A_train.T @ b_train
            x = x - eps * grad
            err_train = err_fun_train(x)
            err_test = err_fun_test(x)
            err_graph_train.append(err_train)
            err_graph_test.append(err_test)
            if np.linalg.norm(grad) < delta:
                print(f'GD converged after {i} iterations')
                break

        acc_train_err.append(err_graph_train)
        acc_test_err.append(err_graph_test)

    avg_train_err, min_train_err = average_and_minimum_graphs(acc_train_err)
    avg_test_err, min_test_err = average_and_minimum_graphs(acc_test_err)

    # Avg
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_err, label='Average Train Error')
    plt.plot(avg_test_err, label='Average Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Average Train and Test Error Plot')
    plt.legend()
    plt.show()

    # Min
    plt.figure(figsize=(10, 5))
    plt.plot(min_train_err, label='Minimum Train Error')
    plt.plot(min_test_err, label='Minimum Test Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (|Ax - b|^2)')
    plt.title('Minimum Train and Test Error Plot')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    diabetes = load_diabetes()
    print(diabetes.data.shape)  # 442 * 10
    print(diabetes.target.shape)  # 442

    task1()
    # task1GPT()
    task2()
    # task2GPT(20000,1e-4, 1e-15)
    task3()
    # task3GPT()
