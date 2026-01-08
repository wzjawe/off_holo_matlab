import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# =============================
# 1. 物理参数
# =============================
N = 512                     # 图像尺寸
lam = 632.8e-9              # 波长 (m)
k = 2 * np.pi / lam         # 波数
dx = dy = 6.5e-6            # 像元尺寸 (m)

theta_x = np.deg2rad(1.5)   # x 方向偏离角
theta_y = np.deg2rad(1.5)   # y 方向偏离角

# =============================
# 2. 空间坐标
# =============================
x = (np.arange(N) - N/2) * dx
y = (np.arange(N) - N/2) * dy
X, Y = np.meshgrid(x, y)
#    显示相位图像
# 1. 矩形台阶相位生成
def generate_rect_step_phase(width, height, radius=60, phase_inside=2*np.pi, phase_outside=0):

    x = (np.arange(N) - N/2) * dx
    y = (np.arange(N) - N/2) * dy
    X, Y = np.meshgrid(x, y)
    # 创建一个圆形台阶
    return np.where((np.abs(X) < 0.8e-3) & (np.abs(Y) < 0.6e-3), phase_inside, 0.0)
    # 创建一个圆形台阶
#np.where((np.abs(X) < 0.8e-3) & (np.abs(Y) < 0.6e-3), phase_step, 0.0)

# 2. 圆形台阶相位生成
# np.where(X**2 + Y**2 <= r**2, phase_step, 0.0)
def generate_circular_step_phase(width, height, radius=60, phase_inside=2*np.pi, phase_outside=0):
    # 创建一个全零的相位图像
    phase = np.zeros((height, width))
    # 计算图像中心坐标
    center_x, center_y = width // 2, height // 2

    # 创建一个圆形台阶
    Y, X = np.ogrid[:height, :width]
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2

    # 对圆内和圆外部分分别赋予不同的相位值
    phase[mask] = phase_inside
    phase[~mask] = phase_outside

    return phase

# 3. Y形台阶相位生成
def generate_y_shape_step_phase(width, height, arm_length=100, angle=30, phase_step=2*np.pi):
    """生成Y形相位图案
    
    参数:
        width, height: 图像尺寸
        arm_length: Y形臂的长度
        angle: Y形分叉的角度（度）
        phase_step: 相位值
    """
    # 创建一个全零的相位图像
    phase = np.zeros((height, width))
    
    # Y形的中心点
    center_x, center_y = width // 2, height // 2
    
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    
    # Y形的三条臂
    # 1. 垂直向上的臂  修改：从减i改为加i，使臂向下延伸
    for i in range(arm_length):
        y = center_y + i
        if 0 <= y < height:
            phase[y, center_x] = phase_step
    
    # 2. 左上臂
    for i in range(arm_length):
        x = int(center_x - i * np.sin(angle_rad))
        y = int(center_y - i * np.cos(angle_rad))
        if 0 <= x < width and 0 <= y < height:
            phase[y, x] = phase_step
    
    # 3. 右上臂
    for i in range(arm_length):
        x = int(center_x + i * np.sin(angle_rad))
        y = int(center_y - i * np.cos(angle_rad))
        if 0 <= x < width and 0 <= y < height:
            phase[y, x] = phase_step
    
    # 对Y形进行加粗处理
    kernel_size = 3  # 加粗的程度
    from scipy.ndimage import binary_dilation
    phase = binary_dilation(phase, iterations=kernel_size).astype(float) * phase_step
    
    return phase

# 4. Peak函数相位生成
def generate_peak_function_phase(width, height, peak_center=(128, 128), peak_height=2*np.pi, spread=20):
    # 创建一个高斯分布的相位函数
    phase = np.zeros((height, width))
    
    # 创建高斯函数
    x = np.linspace(-spread, spread, width)
    y = np.linspace(-spread, spread, height)
    X, Y = np.meshgrid(x, y)
    
    # 高斯分布形成的局部峰值
    peak = peak_height * np.exp(-(X**2 + Y**2) / (2 * spread**2))
    phase += peak
    
    return phase

def generate_phase_image(width, height, frequency=0.5):
    # 生成一个简单的平面波相位图像
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    phase = np.sin(frequency * X + frequency * Y)  # 模拟平面波的相位
    return phase

def generate_spherical_phase(width, height, radius=50, center=(128, 128)):
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # 生成球面波的相位图像
    phase = np.angle(np.exp(1j * Z))  # 使用球面波的相位
    return phase


def add_speckle_noise(phase, mean=0, std=0.1):
    """添加散斑噪声到相位图像中
        mean 是噪声的均值，通常为0。
        std 是噪声的标准差，控制噪声的强度。
        phase 是理想的相位图像
    """
    # 生成均值为0，标准差为std的高斯噪声
    noise = np.random.normal(mean, std, phase.shape)
    
    # 乘性噪声：将噪声加到相位上 形成散斑噪声
    speckle_noise = phase * (1 + noise)
    
    return speckle_noise

# 生成不同形状的相位图像
def generate_phase_images(width=512, height=512):
    # 矩形台阶相位
    rectangular_step_phase = np.where((np.abs(X) < 0.8e-3) & (np.abs(Y) < 0.6e-3), 2, 0.0)

    # 圆形台阶相位
    circular_step_phase = generate_circular_step_phase(width, height, radius=80)

    # Y形台阶相位
    y_shape_step_phase = generate_y_shape_step_phase(width, height, arm_length=100, angle=30)

    # Peak函数相位
    peak_phase = generate_peak_function_phase(width, height)

    # 球面相位
    spherical_phase = generate_spherical_phase(width, height, radius=50)

    # 平面波相位
    plane_wave_phase = generate_phase_image(width, height, frequency=0.5)
    
    return rectangular_step_phase, circular_step_phase, y_shape_step_phase, peak_phase, spherical_phase, plane_wave_phase


# 展示不同形状的相位图像
def display_phase_images():
    rectangular_step_phase, circular_step_phase, y_shape_step_phase, peak_phase, spherical_phase, plane_wave_phase = generate_phase_images()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(rectangular_step_phase, cmap='jet')
    axes[0, 0].set_title("Rectangular Step Phase")
    axes[0, 1].imshow(circular_step_phase, cmap='jet')
    axes[0, 1].set_title("Circular Step Phase")
    axes[1, 0].imshow(y_shape_step_phase, cmap='jet')
    axes[1, 0].set_title("Y Shape Step Phase")
    axes[1, 1].imshow(peak_phase, cmap='jet')
    axes[1, 1].set_title("Peak Function Phase")
    axes[0, 2].imshow(spherical_phase, cmap='jet')
    axes[0, 2].set_title("Spherical Phase")
    axes[1, 2].imshow(plane_wave_phase, cmap='jet')
    axes[1, 2].set_title("Plane Wave Phase")
    
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# 理想复反射场
def phase_to_ideal_reflection_field(phase, amplitude=1.0):
    """
    phase: 理想相位 φ(x,y)
    amplitude: 标量或与 phase 同尺寸的幅值
    amp_std = 0.05
    A = 1 + np.random.normal(0, amp_std, phase.shape)
    A = np.clip(A, 0.5, 1.5)   # 防止物理上不合理
    """
    return amplitude * np.exp(1j * phase)

# 调用展示函数
display_phase_images()
rectangular_step_phase, circular_step_phase, y_shape_step_phase, peak_phase, spherical_phase, plane_wave_phase = generate_phase_images()
U_rect = phase_to_ideal_reflection_field(rectangular_step_phase)
U_circ = phase_to_ideal_reflection_field(circular_step_phase)
U_y    = phase_to_ideal_reflection_field(y_shape_step_phase)
U_peak = phase_to_ideal_reflection_field(peak_phase)
U_sph  = phase_to_ideal_reflection_field(spherical_phase)   
U_plan = phase_to_ideal_reflection_field(plane_wave_phase)
# 参考光函数
 

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(np.abs(U_rect), cmap='gray')
plt.title('Rectangular Step Phase')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(np.abs(U_circ), cmap='gray')
plt.title('Circular Step Phase')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(np.abs(U_y), cmap='gray')
plt.title('Y Shape Step Phase')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(np.abs(U_peak), cmap='gray')
plt.title('Peak Function Phase')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(np.abs(U_sph), cmap='gray')
plt.title('Spherical Phase')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(np.abs(U_plan), cmap='gray')
plt.title('Plane Wave Phase')
plt.axis('off')

plt.tight_layout()
plt.show()