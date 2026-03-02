import numpy as np

A = np.array([
    [5.9, 1.2, 2.1, 0.9],
    [1.2, 7.2, 1.5, 2.5],
    [2.1, 1.5, 9.8, 1.3],
    [0.9, 2.5, 1.3, 6.1]
], float)

b = np.array([-2.0, 5.3, 10.3, 12.6], float)


# ---------- Гаусс с выбором главного элемента ----------
def gauss_pivot(A, b):
    A = A.copy()
    b = b.copy()
    n = len(b)
    for k in range(n - 1):
        p = np.argmax(abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x, A


xg, U = gauss_pivot(A, b)


# ---------- Зейдель с записью итераций ----------
def seidel_with_table(A, b, eps=0.1, max_iter=10000):
    n = len(b)
    x = np.zeros(n)

    # Для хранения истории итераций
    iteration_numbers = []
    x_values = []
    errors = []

    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = A[i, :i] @ x_new[:i]
            s2 = A[i, i + 1:] @ x[i + 1:]
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        error = np.linalg.norm(x_new - x, np.inf)

        # Сохраняем данные итерации
        iteration_numbers.append(k + 1)
        x_values.append(x_new.copy())
        errors.append(error)

        if error < eps:
            return x_new, k + 1, error, iteration_numbers, x_values, errors

        x = x_new

    return x_new, max_iter, error, iteration_numbers, x_values, errors
xs, it, final_error, iter_nums, x_vals, err_vals = seidel_with_table(A, b, eps=0.1)

# Вычисляем абсолютную погрешность относительно метода Гаусса
abs_errors = np.abs(xg - xs)
max_abs_error = np.max(abs_errors)

print("=" * 80)
print("МЕТОД ГАУССА С ВЫБОРОМ ГЛАВНОГО ЭЛЕМЕНТА")
print("=" * 80)
print(f"Решение (считаем точным): {xg}")
print()

print("=" * 80)
print("МЕТОД ЗЕЙДЕЛЯ (точность ε < 0.1)")
print("=" * 80)
print(f"Решение: {xs}")
print(f"Количество итераций: {it}")
print(f"Конечная погрешность (max|x^{it} - x^{it - 1}|): {final_error:.6f}")
print(f"Достигнута ли точность < 0.1: {'Да' if final_error < 0.1 else 'Нет'}")
print()

print("=" * 80)
print("АБСОЛЮТНАЯ ПОГРЕШНОСТЬ МЕТОДА ЗЕЙДЕЛЯ")
print("=" * 80)
print(f"Максимальная абсолютная погрешность относительно метода Гаусса: {max_abs_error:.6f}")
print(f"Абсолютные погрешности по компонентам:")
for i in range(4):
    print(f"  Δx{i+1} = |{xg[i]:.6f} - {xs[i]:.6f}| = {abs_errors[i]:.6f}")
print()

print("=" * 80)
print("ТАБЛИЦА ИТЕРАЦИЙ МЕТОДА ЗЕЙДЕЛЯ")
print("=" * 80)

# Форматирование заголовка таблицы
header = f"{'Итер.':<6} {'x1':<15} {'x2':<15} {'x3':<15} {'x4':<15} {'Погрешность ε':<15}"
print(header)
print("-" * 80)

# Выводим все итерации
for i in range(it):
    x = x_vals[i]
    row = f"{iter_nums[i]:<6} {x[0]:<15.1f} {x[1]:<15.1f} {x[2]:<15.1f} {x[3]:<15.1f} {err_vals[i]:<15.1f}"
    print(row)

# Проверка условия сходимости
print("\n" + "=" * 80)
print("ПРОВЕРКА ДОСТАТОЧНОГО УСЛОВИЯ СХОДИМОСТИ МЕТОДА ЗЕЙДЕЛЯ")
print("=" * 80)
print("Условие: |a_ii| > Σ|a_ij| для всех i, где j ≠ i")
print("(модуль диагонального элемента > суммы модулей остальных элементов строки)")
print()

# Проверка строгого диагонального преобладания по строкам
print("Вычисляем для каждой строки:")
print("-" * 50)

condition_met = True
for i in range(4):
    diag_element = abs(A[i, i])

    # Сумма модулей остальных элементов строки
    sum_other = 0
    other_elements = []
    for j in range(4):
        if j != i:
            sum_other += abs(A[i, j])
            other_elements.append(f"|{A[i, j]:.1f}|")

    # Проверяем условие
    is_greater = diag_element > sum_other
    condition_symbol = "✓" if is_greater else "✗"

    print(f"Строка {i + 1}:")
    print(f"  Диагональный элемент: |a_{i + 1}{i + 1}| = |{A[i, i]:.1f}| = {diag_element:.1f}")
    print(f"  Остальные элементы: {' + '.join(other_elements)} = {sum_other:.1f}")
    print(f"  Проверка: {diag_element:.1f} > {sum_other:.1f} ? {condition_symbol}")

    if not is_greater:
        condition_met = False

    if i < 3:
        print("-" * 40)

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТ ПРОВЕРКИ:")
print("=" * 80)

if condition_met:
    print("✓ Условие строгого диагонального преобладания по строкам ВЫПОЛНЕНО")
    print("  Метод Зейделя гарантированно сходится к решению")
else:
    print("✗ Условие строгого диагонального преобладания по строкам НЕ ВЫПОЛНЕНО")
    print("  Метод Зейделя может не сходиться (нужна дополнительная проверка)")

# Вывод матрицы преобразований
print("\n" + "=" * 80)
print("МАТРИЦА ПРЕОБРАЗОВАНИЙ (верхняя треугольная):")
print("=" * 80)
print("(форматировано с точностью до 2 знаков после запятой)")
np.set_printoptions(precision=2, suppress=True)
print(U)
