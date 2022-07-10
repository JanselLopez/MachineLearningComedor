import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, style
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    '''
    getting the values
    '''
    DATASET = pd.read_csv('Comedor.csv', sep=',')
    x1, x2, y = DATASET['week_day'], DATASET['food_value'], DATASET['how_many_people']
    df = pd.DataFrame(DATASET)
    '''
    model creation
    '''
    lin_reg = LinearRegression()
    lin_reg.fit(df[['week_day', 'food_value']], df['how_many_people'])
    '''
    for represent the plane
    '''
    X, Y = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x1), max(x2), 10))
    z = lin_reg.intercept_ + lin_reg.coef_[0] * X + lin_reg.coef_[1] * Y
    '''
    macking the prediction
    '''
    x_new = np.array([[6, 14000]])
    people = lin_reg.predict(x_new)
    print(int(people[0]))
    '''
    representing the function in 3d
    '''
    ax = plt.axes(projection='3d')
    ax.scatter3D(x1, x2, y, color='red')
    ax.plot_surface(X, Y, z, alpha=0.75)
    plt.xlabel('week_day')
    plt.ylabel('food_value')
    plt.show()
