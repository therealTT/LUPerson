from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import cv2
import sys
#from get_path import get_image_path_by_index
from natsort import natsorted
import os 

import pickle

mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

# meta_data = np.load('/home/niranjan/projects/LUPerson/fast-reid/resnet50.npz', allow_pickle=True)
# print(meta_data.keys(),"keys")
# print(meta_data)
# sim_matrix = meta_data['arr_0']
# print(type(sim_matrix))
#print(sim_matrix)
with open ('/home/niranjan/projects/person_reid_evaluation/LUPerson/fast-reid/ourmodel_simmtx.pkl','rb') as fp:
    meta_data = pickle.load(fp)
print(meta_data.keys())
sim_matrix = meta_data['sim_mtx']
sim_matrix = (sim_matrix - sim_matrix.min())/(sim_matrix.max()-sim_matrix.min())
img_paths = meta_data['img_paths']
labels = meta_data['labels']
query_imgs = img_paths[:3368]
gallery_imgs = img_paths[3368:]
query_labels = labels[:3368]
gallery_labels = labels[3368:]
gallery_labels = [item.item() for item in gallery_labels]
query_labels = [item.item() for item in query_labels]
galley_enumerated_list = list(enumerate(gallery_labels))
query_enumerated_list = list(enumerate(query_labels))
# Sort the enumerated list by values
gallery_sorted_enumerated_list = sorted(galley_enumerated_list, key=lambda x: x[1])
query_sorted_enumerated_list = sorted(query_enumerated_list, key=lambda x: x[1])

# Extract the sorted indexes from the sorted enumerated list
gallery_sorted_indexes = [item[0] for item in gallery_sorted_enumerated_list]
query_sorted_indexes = [item[0] for item in query_sorted_enumerated_list]
# Sort the original list in place
gallery_labels.sort()
gallery_labels = gallery_labels[2798:]
query_labels.sort()
gallery_imgs = [gallery_imgs[i] for i in gallery_sorted_indexes]
gallery_imgs = gallery_imgs[2798:]
query_imgs = [query_imgs[i] for i in query_sorted_indexes]
# Now, 'original_list' is sorted, and 'sorted_indexes' contains the new indexes
#print("Sorted List:", gallery_labels)
print(len(gallery_labels))
print(len(set(gallery_labels)))
print(len(set(query_labels)))
print(sim_matrix[426,7],"score")
sim_matrix = sim_matrix[query_sorted_indexes][:, gallery_sorted_indexes]
sim_matrix = sim_matrix[:,2798:]
print(sim_matrix[0,0],"score")
gallery_count = []
query_count = []
for gallery_id in set(gallery_labels[:500]):
    gallery_count.append(gallery_labels.count(gallery_id))
#print(gallery_count)
print(len(gallery_count))
print(gallery_count[0])
for query_id in set(query_labels[:500]):
    query_count.append(query_labels.count(query_id))
#print(query_count)
print(len(query_count))
print(query_count[0])

print(len(query_imgs),len(gallery_imgs),len(query_labels),len(gallery_labels))
img1 = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
img2 = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
# img1 = (((imgs_sorted[0] * std) + mean) * 255.0).astype(np.uint8)
# img2 = (((imgs_sorted[185] * std) + mean) * 255.0).astype(np.uint8)

app = Dash(__name__)

hmap1 = px.imshow(sim_matrix[:500,:500], text_auto=True)
# hmap2 = px.imshow(sim_matrix[3001:, 3001:], text_auto=True)
hmap1.layout.height = 1800
# hmap2.layout.height = 1800
hmap1.layout.width = 1800
# hmap2.layout.width = 1800

# hmap1.write_html('hmap1.html')
# hmap2.write_html('hmap2.html')
# hmap1 = go.Figure(data=go.Heatmap(z=sim_matrix[:6000, :6000]))
# hmap1.show()

# dt_fol = '/home/suyash/face_reid/datasets/imfdb_new/'
# directories = natsorted(os.listdir(dt_fol))
# print(directories)
# fol_length = [(len(os.listdir(dt_fol +i))) for i in directories]
# print(fol_length)

# j = 0
# for i in fol_length:  
#     j+=i
#     hmap1.add_vline(
#         x=j-1, line_width=5, line_dash="solid",
#         line_color="white")
#     hmap1.add_hline(
#         y=j-1, line_width=5, line_dash="solid",
#         line_color="white")
#     break
gal = 0
count = 0
for gal_co in gallery_count:
    count+=1
    # print(gal_co, "gallery", count)
    gal+=gal_co
    hmap1.add_vline(
        x=gal-1, line_width=2, line_dash="solid",
        line_color="white")
quer = 0
count=0
for quer_co in query_count:
    count+=1
    # print(quer_co, "query", count)
    quer += quer_co
    hmap1.add_hline(
        y=quer-1, line_width=2, line_dash="solid",
        line_color="white")

app.layout = html.Div([
    html.H1('ReID Model Evaluation'),
    dcc.Graph(figure=hmap1, id="heatmap1"),
    html.Br(),
    # dcc.Graph(figure=hmap2, id="heatmap2"),
    html.Br(),
    html.H1('Person ID : None              Distance : 0.0', id='h41'),
    dcc.Graph(figure=px.imshow(img1), id='img1'),
    html.H1('Person ID : None              Distance : 0.0', id='h42'),
    html.Br(),
    dcc.Graph(figure=px.imshow(img2), id='img2'),
])

def apply_inverse_transform(img):
    # return (((img * std) + mean) * 255.0 * 255.0).astype(np.uint8)
    return (((img * std) + mean) * 255.0).astype(np.uint8)

@app.callback(
    # Output("Heatmap", "figure"),
    Output("img1", "figure"),
    Output("img2", "figure"),
    Output("h41", "children"),
    Output("h42", "children"),
    Input("heatmap1", "clickData"))
def filter_heatmap(clickData):
    print(clickData)
    # clickData = None
    if clickData is not None:
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        print(x,y)
        # print(query_imgs[x])
        # print(gallery_imgs[y])
        # img_path_1, fol_index_1 = get_image_path_by_index(x)
        # img_path_2, fol_index_2 = get_image_path_by_index(y)
        img_path_1 = query_imgs[y]
        img_path_2 = gallery_imgs[x]
        fol_index_1 = query_labels[y]
        fol_index_2 = gallery_labels[x]
        label1 = f'Person ID : {fol_index_1}              Distance : {sim_matrix[y, x]}'
        label2 = f'Person ID : {fol_index_2}              Distance : {sim_matrix[y, x]}'
        img_r1 = px.imshow(cv2.cvtColor(cv2.imread(img_path_1), cv2.COLOR_BGR2RGB))
        img_r2 = px.imshow(cv2.cvtColor(cv2.imread(img_path_2), cv2.COLOR_BGR2RGB))
    else:
        img_r1 = px.imshow(np.random.randint(0, 255, (112, 112, 3)).astype(np.uint8))
        img_r2 = px.imshow(np.random.randint(0, 255, (112, 112, 3)).astype(np.uint8))
        label1 = 'Person ID : None              Distance : 0.0'
        label2 = 'Person ID : None              Distance : 0.0'
    return img_r1, img_r2, label1, label2

app.run_server(debug=True)