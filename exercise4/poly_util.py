import pandas as pd
import numpy as np
import cv2
# from geometry_msgs.msg import Polygon, Point32
from pathlib import Path
import math
import matplotlib.pyplot as plt
import scipy.interpolate as interp


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


# def pandas_to_ros(df):
#     poly = Polygon()
#     for point in df.values:
#         p = Point32()
#         p.x = point[0]
#         p.y = point[1]
#         p.z = point[2]
#         poly.points.append(p)
#     return poly


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
def interpolate(df, md, method):
    df = reindex(df, md)
    return df.interpolate(method=method, axis=0)


def cluster(df):
    pass


def save(df, path):
    df.to_csv(path)


def load(path):
    return pd.read_csv(path)


def load_polygons(path):
    p = Path(path)
    polygons = []
    names = []
    for csv in [f for f in p.iterdir()]:
        df = pd.read_csv(csv)
        polygons.append(df)
        names.append(csv.stem)
    return polygons, names


def draw_polygons(polys, names, bbox, border, width, height):
    image = np.zeros((height, width, 3), np.uint8)
    minx, miny, maxx, maxy = bbox

    def mapy(y):
        return round((y - miny) / (maxy - miny) * ((width - border) - (0 + border)) + (0 + border))

    def mapx(x):
        return round((x - minx) / (maxx - minx) * ((height - border) - (0 + border)) + (0 + border))

    assert (len(names) == len(polys))
    for i in range(len(polys)):
        df = polys[i]
        name = names[i]
        n = name.split('_')[1]
        xl = (list(map(mapx, df['x'].tolist())))
        yl = (list(map(mapy, df['y'].tolist())))
        xl = np.array(xl, dtype=np.int32)
        yl = np.array(yl, dtype=np.int32)
        assert (len(xl) == len(yl))
        polygon = []
        for j in range(len(xl)):
            polygon.append([xl[j], yl[j]])
        polygon = np.array(polygon, dtype=np.int32)
        c = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if n == '20':
            c = (0, 0, 255)
            #
        # if n == '19':
        #     c = (0, 255, 255)
        # for j in range(len(polygon)):
        #     cv2.putText(image, str(j), (polygon[j][0], polygon[j][1]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            cv2.fillPoly(image, [polygon], c)
            # pass
        except Exception as e:
            print(e)
    cv2.imshow('test', image)


def fix_intersections(polys, names):
    for df, n in zip(polys, names):
        nn = 'poly_' + n.split('_')[1] + '.csv'
        df = interpolate(df, 0.08, 'piecewise_polynomial')
        # plt.plot(df['x'].values, df['y'].values, '.', df['x'].values, df['y'].values, '--')
        # plt.show()
        df.to_csv('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons/' + nn)


def fix_polys(polys, names):
    p2c = {'16_4': 2}
    p1c = {'17_5': 2}
    r1, r2, r3 = [], [], []
    for p, name in zip(polys, names):
        n = name.split('_')[1] + '_' + name.split('_')[2]
        n1 = int(n.split('_')[1])
        px, py, pz, v = p['x'].values, p['y'].values, p['z'].values, p.index.values
        if n in p1c:
            for i in range(len(px)):
                cx, cy, cz, cv = px[i], py[i], pz[i], v[i]
                if i+1 <= p1c[n]:
                    r1.append((cv + n1*100, cx, cy, cz))
                else:
                    r3.append((cv - n1*100, cx, cy, cz))
        elif n in p2c:
            for i in range(len(px)):
                cx, cy, cz, cv = px[i], py[i], pz[i], v[i]
                if i+1 <= p2c[n]:
                    pass # r3.append((cv + n1, cx, cy, cz))
                else:
                    r2.append((cv - n1*100, cx, cy, cz))
    r1 = sorted(r1, key=lambda x: x[0])
    r2 = sorted(r2, key=lambda x: x[0])
    r3 = sorted(r3, key=lambda x: x[0])

    v1, r1x, r1y, r1z = zip(*iter(r1))
    v2, r2x, r2y, r2z = zip(*iter(r2))
    v3, r3x, r3y, r3z = zip(*iter(r3))
    v1, r1x, r1y, r1z = list(v1), list(r1x), list(r1y), list(r1z)
    v2, r2x, r2y, r2z = list(v2), list(r2x), list(r2y), list(r2z)
    v3, r3x, r3y, r3z = list(v3), list(r3x), list(r3y), list(r3z)

    rx = r1x + r3x
    rx.append(rx[0])
    ry = r1y + r3y
    ry.append(ry[0])
    rz = r1z + r3z
    rz.append(rz[0])
    rx2 = list(reversed(r2x)) + r3x
    rx2.append(rx2[0])
    ry2 = list(reversed(r2y)) + r3y
    ry2.append(ry2[0])
    rz2 = list(reversed(r2z)) + r3z
    rz2.append(rz2[0])
    df = pd.DataFrame(data={'x': rx, 'y': ry, 'z': rz})
    df2 = pd.DataFrame(data={'x': rx2, 'y': ry2, 'z': rz2})
    df = interpolate(df, 0.08, 'piecewise_polynomial')
    df2 = interpolate(df2, 0.08, 'piecewise_polynomial')
    #
    plt.plot(df['x'].values, df['y'].values, '.', df2['x'].values, df2['y'].values, '.')
    plt.show()
    df.to_csv('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons/poly_16.csv')
    df2.to_csv('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons/poly_17.csv')


def clean_polygons(polys):
    for df in polys:
        for index, row in df.iterrows():
            if not row['Unnamed: 0'] == 0 and (-0.000001 < row['Unnamed: 0'] % 0.1 < 0.00001 or 0.09999999 < row['Unnamed: 0'] % 0.1):
                df.drop(index, inplace=True)
    return polys


def df_graph(polys, names, k):
    for df, n in zip(polys, names):
        if int(n.split('_')[1]) in k:
            print(k)
            x = df['x'].values
            y = df['y'].values
            plt.plot(x, y, '.')
            plt.show()


def save_base(polys, names):
    path = Path('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons_base/')
    for p, n in zip(polys, names):
        npath = path.joinpath(n + '.csv')
        p.to_csv(npath)


def main():
    polys, names = load_polygons('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons/')
    # polys, names = load_polygons('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/pbi/')
    # polys = polys + polys2
    # names = names + names2
    #polys, names = load_polygons('C:/Users/eirik/PycharmProjects/d4dl2/exercise4/polygons_base/')
    minx, miny, maxx, maxy = None, None, None, None
    for poly in polys:
        maxx = poly['x'].max() if (maxx is None or poly['x'].max() > maxx) else maxx
        minx = poly['x'].min() if (minx is None or poly['x'].min() < minx) else minx
        maxy = poly['y'].max() if (maxy is None or poly['y'].max() > maxy) else maxy
        miny = poly['y'].min() if (miny is None or poly['y'].min() < miny) else miny
    # clean_polygons(polys)
    # save_base(polys, names)
    # df_graph(polys, names, [19])
    # fix_polys(polys, names)
    draw_polygons(polys, names, (minx, miny, maxx, maxy), 30, 1024, 1024)
    # fix_intersections(polys, names)


if __name__ == '__main__':
    main()
