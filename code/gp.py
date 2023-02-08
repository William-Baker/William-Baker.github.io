#%%

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import numpy as np
import matplotlib.pyplot as plt


y = np.array([4105.35, 4119.57, 4136.69, 4158.68, 4070.07, 4020.85, 4049.27, 4053.72, 4036.08, 3982.71, 4001.74, 3978.14, 3909.04, 3911.84, 4002.25, 3999.28, 3960.60, 3977.57, 3932.35, 3888.57, 3910.82, 3823.37, 3839.74, 3840.36, 3853.29, 3829.06, 3805.45, 3829.56, 3843.34, 3815.11, 3853.26, 3839.49, 3810.47, 3853.79, 3890.91, 3958.37, 4015.54, 4069.38, 3939.29, 3954.17, 3947.79, 3933.28, 3996.63, 4052.02, 4040.17, 4087.14, 3957.18, 3964.19, 4005.36, 4023.34, 4000.30, 3965.51, 3956.23, 3966.39, 3919.26, 3976.82, 4006.41, 3977.97, 3963.72, 3859.89, 3810.94, 3817.02, 3780.71, 3766.98, 3733.25, 3852.90, 3901.79, 3881.85, 3808.26, 3834.69, 3825.97, 3799.44, 3762.01, 3657.10, 3689.05, 3703.11, 3746.26, 3638.65, 3690.41, 3520.37, 3590.83, 3595.86, 3647.51, 3706.74, 3771.97, 3753.25, 3726.46, 3609.78, 3633.48, 3687.01, 3651.94, 3686.44, 3682.72, 3727.14, 3782.36, 3871.40, 3875.23, 3849.91, 3880.95, 3932.41, 3940.73, 4037.12, 4083.67, 4022.94, 3959.94, 3909.43, 3930.89, 3994.66, 3936.73, 4000.67, 4041.25, 4034.58, 4198.74, 4153.26, 4126.55, 4133.09, 4195.08, 4266.31, 4273.13, 4280.40, 4290.46, 4269.37, 4225.02, 4227.40, 4181.02, 4133.11, 4155.93, 4115.87, 4154.85, 4107.96, 4104.21, 4112.38, 4087.33, 4026.13, 3951.43, 3953.22, 3965.72, 3998.43, 3955.47, 3935.32, 3860.73, 3883.79, 3818.00, 3763.99, 3779.67, 3851.95, 3880.94, 3888.26, 3858.85, 3831.98, 3792.61, 3781.00, 3785.99, 3825.09, 3913.00, 3920.76, 3821.75, 3774.71, 3733.89, 3715.31, 3665.90, 3728.18, 3764.05, 3763.52, 3838.15, 3974.39, 4101.65, 4147.12, 4096.47, 4134.72, 4137.57, 4095.41, 4149.78, 4151.09, 4077.43, 3984.60, 3929.59, 3942.94, 3919.42, 3927.76, 3899.00, 4051.98, 4052.00, 4013.02, 3963.90, 3903.95, 3990.08, 4035.18, 4081.27, 4128.17, 4270.43, 4181.18, 4159.78, 4130.61, 4253.75, 4222.58, 4186.52, 4278.14, 4255.34, 4385.83, 4489.17, 4472.26, 4390.63, 4385.63, 4449.12, 4394.30, 4437.59, 4462.64, 4494.15, 4474.65, 4494.17, 4572.45, 4547.97, 4540.32, 4599.02, 4624.20, 4602.86, 4541.09, 4522.91, 4469.98, 4493.10, 4469.10, 4462.40, 4407.34, 4345.11, 4288.14, 4188.82, 4202.75, 4279.50, 4252.55, 4223.10, 4202.66, 4327.01, 4342.12, 4401.31, 4322.56, 4363.14, 4354.17, 4298.38, 4155.77, 4324.93, 4332.74, 4384.57, 4456.06, 4455.75, 4429.28, 4412.61, 4506.27, 4553.24, 4547.00, 4480.02, 4505.75])
y = y / y.max()
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
X = np.linspace(0, 365, y.shape[0]).reshape(-1, 1)
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=10, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")
#%%

import plotly.graph_objects as go

fig = go.Figure()



upper = mean_prediction + 1.96 * std_prediction
lower = mean_prediction - 1.96 * std_prediction
fig.add_trace(go.Scatter(
    x=np.concatenate([X.ravel(), X.ravel()[::-1]]), 
    y=np.concatenate([upper, lower[::-1]]),
    fill='toself',
    #mode='lines',
    #line_color='indigo',
    name='3σ',
    mode= 'none',
     fillcolor='rgba(0,150,0,0.2)',
     hoverinfo='none',   
    ))

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)


fig.add_trace(go.Scatter(x=X.ravel(), y=mean_prediction, name='Prediction', mode='lines',hoverinfo='none', marker=dict(color='#00CC96')))
fig.add_trace(go.Scatter(x=X.ravel(), y=y, name='Signal', marker=dict(color='#636EFA')))
fig.add_trace(go.Scatter(x=X_train.ravel(), y=y_train, name='Signal Observations', mode='markers', marker_symbol='x', marker=dict(size=10, color='#EF553B')))

fig.write_html('gp_chart.html', include_plotlyjs=False)


# %%