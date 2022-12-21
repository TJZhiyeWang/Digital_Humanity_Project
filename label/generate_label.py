import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PATH = "../inventory_sgi1931_r2022/SGI_1931.shp"
seg = gpd.read_file(PATH)
#EPSG:21781 x-y range
x_min = 619741.59
y_min = 170763.96

x_max = 673471.64
y_max = 132341.12
x_gap = x_max - x_min
y_gap = y_max - y_min
print("x_gap: ",x_gap )
print("y_gap: ",y_gap )

labels = np.load("./labels_small.npy")
y_len, x_len = labels.shape
print("x_len: ",x_len )
print("y_len: ",y_len )

x_zoom = x_gap/x_len
y_zoom = y_gap/y_len

x_list = []
y_list = []
points = []

end_point = 0

for i in tqdm(range(end_point, x_len)):
    for j in range(y_len):
        x_list.append(i)
        y_list.append(j)
        points.append(Point(i * x_zoom + x_min, y_min + j * y_zoom))
        if j % 3000 == 0 or j == y_len - 1:
            res = gpd.GeoDataFrame({'geometry': points, 'x':x_list, 'y':y_list}).sjoin(seg, how="inner", predicate='intersects')
            for _, (x,y) in enumerate(zip(res.x, res.y)):
                labels[y,x] = True
            x_list = []
            y_list = []
            points = []
    # if i % 500 == 0:
    #     np.save("./new_labels_small.npy", labels)

np.save("./new_labels_small_1931.npy", labels)