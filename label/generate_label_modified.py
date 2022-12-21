import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import warnings
import xarray as xr

warnings.filterwarnings("ignore")


#获取仿射矩阵信息
def Getgeotrans():
    dataset = xr.open_rasterio('./1931_modified.tif')
    return dataset.transform

def pixel2Coord(Xpixel,Ypixel,GeoTransform):
    XGeo = GeoTransform[0]+GeoTransform[1]*Xpixel+Ypixel*GeoTransform[2]
    YGeo = GeoTransform[3]+GeoTransform[4]*Xpixel+Ypixel*GeoTransform[5]
    return XGeo,YGeo

PATH = "../inventory_sgi1931_r2022/SGI_1931.shp"
seg = gpd.read_file(PATH)

labels = np.load("./mod_labels_small.npy")
y_len, x_len = labels.shape
print("x_len: ",x_len )
print("y_len: ",y_len )

urbanData = xr.open_rasterio('./1931_modified.tif')
ur  = xr.DataArray(urbanData, name='myData')
ur  = ur.to_dataframe().reset_index() 
ur = ur[ur['band'] == 1]
x_l = ur['x']
y_l = ur['y']


x_list = []
y_list = []
points = []

end_point = 0

for i in tqdm(range(end_point, x_len)):
    for j in range(y_len):
        x_list.append(i)
        y_list.append(j)
        points.append(Point(x_l[i + j * x_len], y_l[i + j * x_len]))
        if j % 3000 == 0 or j == y_len - 1:
            res = gpd.GeoDataFrame({'geometry': points, 'x':x_list, 'y':y_list}).sjoin(seg, how="inner", predicate='intersects')
            for _, (x,y) in enumerate(zip(res.x, res.y)):
                labels[y,x] = True
            x_list = []
            y_list = []
            points = []
    if i % 500 == 0:
        np.save("./mod_labels_small_1931.npy", labels)

np.save("./mod_labels_small_1931.npy", labels)