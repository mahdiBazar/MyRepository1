import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

#---------------------------------------------------------------
# تولید داده نمونه
X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)


#---------------------------------------------------------------
# تابع هزینه برای بهینه‌سازی
def svm_fitness(params):
    C, gamma = params
    if C <= 0 or gamma <= 0:  # محدودیت‌های مثبت بودن
        return -np.inf
    model = SVC(C=C, kernel='rbf', gamma=gamma)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()



#---------------------------------------------------------------
# الگوریتم PSO
def pso_optimize(num_particles=20, max_iter=50):
    # محدوده مقادیر C و γ
    c_range = [0.1, 100]
    gamma_range = [0.01, 10]

    # مقداردهی اولیه
    particles = np.random.uniform([c_range[0], gamma_range[0]], [c_range[1], gamma_range[1]], size=(num_particles, 2))
    velocities = np.random.uniform(-1, 1, size=(num_particles, 2))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([svm_fitness(p) for p in particles])
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = np.max(personal_best_scores)

    # بهینه‌سازی
    w = 0.5  # وزن اینرسی
    c1 = 1.5  # عامل شناختی
    c2 = 1.5  # عامل اجتماعی

    for iteration in range(max_iter):
        for i, particle in enumerate(particles):
            # به‌روزرسانی سرعت و موقعیت
            velocities[i] = (w * velocities[i]
                             + c1 * np.random.rand() * (personal_best_positions[i] - particle)
                             + c2 * np.random.rand() * (global_best_position - particle))
            particles[i] += velocities[i]

            # محدود کردن ذرات به محدوده مقادیر
            particles[i] = np.clip(particles[i], [c_range[0], gamma_range[0]], [c_range[1], gamma_range[1]])

            # محاسبه تناسب (fitness)
            score = svm_fitness(particles[i])

            # به‌روزرسانی بهترین موقعیت شخصی
            if score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score

        # به‌روزرسانی بهترین موقعیت جهانی
        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
        global_best_score = np.max(personal_best_scores)

        print(f"Iteration {iteration + 1}/{max_iter}, Best Score: {global_best_score:.4f}")

    return global_best_position, global_best_score

# اجرای PSO
best_params, best_score = pso_optimize()
print(f"Best Parameters: C = {best_params[0]:.4f}, gamma = {best_params[1]:.4f}, Best Score = {best_score:.4f}")
