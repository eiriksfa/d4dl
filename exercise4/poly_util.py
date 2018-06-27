import pandas as pd
import numpy as np
from geometry_msgs.msg import Polygon, Point32


def ros_to_np(poly):
    points = []
    for point in poly.points:
        points.append([point.x, point.y, point.z])
    points = np.array(points)
    return points


def ros_to_pd(poly):
    points = ros_to_np(poly)
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return df


def pandas_to_ros(df):
    poly = Polygon()
    for point in df.values:
        p = Point32()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        poly.points.append(p)
    return poly


def reindex(df, md):
    df['distance'] = np.sqrt(
        (df['x'] - df['x'].shift()) ** 2 + (df['y'] - df['y'].shift()) ** 2 + (df['z'] - df['z'].shift()) ** 2)
    df['distance'][0] = 0
    for i in range(1, len(df)):
        d = df.loc[i, 'distance']
        df.loc[i, 'distance'] = df.loc[i - 1, 'distance'] + d
    m = df['distance'].max()
    nindex = np.arange(md, m, md)
    nindex = np.sort(np.concatenate([nindex, df['distance'].values]))
    df = df.set_index('distance')
    df = df.reindex(pd.Index(nindex))
    return df


# TODO: Test
def interpolate(df, md):
    df = reindex(df, md)
    return df.interpolate(method='linear', axis=0)


def cluster(df):
    pass


def save(df, path):
    df.to_csv(path)


def load(path):
    return pd.read_csv(path)


