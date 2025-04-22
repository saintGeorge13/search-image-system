import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from pymilvus import MilvusClient
import glob
import streamlit as st
from tqdm import tqdm
import math

client = MilvusClient("milvus_demo.db")

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/root/autodl-tmp/model")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/root/autodl-tmp/model")

# 定义图片文件夹
IMAGE_DIR = "gpt4o"

# 设置Milvus集合
COLLECTION_NAME = "image_collection"

def setup_milvus():
    """创建Milvus集合"""
    # 检查集合是否存在，如果存在则删除
    if client.has_collection(COLLECTION_NAME):
        # 删除
        client.drop_collection(COLLECTION_NAME)
    
    # 创建集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768,  # CLIP-ViT-Large-Patch14的向量维度为768
        metric_type="COSINE"  # 使用余弦相似度
    )
    print(f"已创建Milvus集合: {COLLECTION_NAME}")

def get_image_embedding(image_path):
    """计算图片的CLIP嵌入向量"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            
        # 获取并归一化向量
        embedding = outputs.detach().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding[0]  # 返回形状为(768,)的一维数组
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

def get_text_embedding(text):
    """计算文本的CLIP嵌入向量"""
    inputs = processor(text=text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    # 获取并归一化向量
    embedding = text_features.detach().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding[0]  # 返回形状为(768,)的一维数组

def index_images():
    """为目录中的所有图片创建索引"""
    # 获取目录中的所有图片
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
                 glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) + \
                 glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not image_files:
        print(f"警告: 在 {IMAGE_DIR} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 计算所有图片的嵌入向量
    embeddings = []
    image_paths = []
    
    for img_path in tqdm(image_files, desc="计算图片向量"):
        embedding = get_image_embedding(img_path)
        if embedding is not None:
            embeddings.append(embedding.tolist())
            image_paths.append(img_path)
    
    # 将嵌入向量插入到Milvus
    entities = [
        {"id": i, "vector": emb, "path": path}
        for i, (emb, path) in enumerate(zip(embeddings, image_paths))
    ]
    
    if entities:
        client.insert(
            collection_name=COLLECTION_NAME, 
            data=entities
        )
        print(f"已将 {len(entities)} 张图片的向量存入Milvus")
    else:
        print("没有有效的图片向量可以存入")

def search_similar_images(text, top_k=5):
    """根据文本查找相似图片"""
    # 计算文本的嵌入向量
    text_embedding = get_text_embedding(text)
    
    # 在Milvus中搜索相似向量
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_embedding.tolist()],
        limit=top_k,
        output_fields=["path"]
    )
    
    if search_results and search_results[0]:
        
        # 显示结果
        similar_images = []
        for i, result in enumerate(search_results[0]):
            similarity = result["distance"]
            path = result["entity"]["path"]
            similar_images.append((path, similarity))
        
        return similar_images
    else:
        return []

def search_similar_images_by_image(image_path, top_k=5):
    """根据图片查找相似图片"""
    # 计算输入图片的嵌入向量
    image_embedding = get_image_embedding(image_path)
    
    if image_embedding is None:
        print(f"无法处理输入图片: {image_path}")
        return []
    
    # 在Milvus中搜索相似向量
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[image_embedding.tolist()],
        limit=top_k+1,  # 多查询一个，因为可能会包含输入图片自身
        output_fields=["path"]
    )
    
    if search_results and search_results[0]:
        # 过滤掉输入图片自身（如果存在于结果中）
        # filtered_results = [r for r in search_results[0] if r["entity"]["path"] != image_path]
        filtered_results = [r for r in search_results[0]]
        # 如果过滤后结果数量仍然超过top_k，则只保留前top_k个
        filtered_results = filtered_results[:top_k]
        
        # 显示结果
        similar_images = []
        for i, result in enumerate(filtered_results):
            similarity = result["distance"]
            path = result["entity"]["path"]
            similar_images.append((path, similarity))
        
        return similar_images
    else:
        return []

if __name__ == "__main__":
    # 设置页面布局为宽屏
    st.set_page_config(layout="wide", page_title="图文匹配搜索系统")
    
    # 添加页面标题
    st.title("图文匹配搜索系统")
    
    # 创建左右两栏布局
    col1, col2 = st.columns([1.5, 3])
    
    with col1:
        st.header("搜索选项")
        # 文本输入框
        text_query = st.text_area("输入文本描述", height=100)
        
        # 图片上传
        uploaded_file = st.file_uploader("或者上传图片", type=["jpg", "jpeg", "png"])
        
        # 设置每次搜索返回的图片数量
        top_k = st.slider("显示结果数量", min_value=5, max_value=30, value=12, step=1)
        
        # 搜索按钮
        if st.button("搜索"):
            if not client.has_collection(COLLECTION_NAME):
                setup_milvus()
                index_images()
            
            # 检查是否已有结果
            if 'search_results' not in st.session_state:
                st.session_state.search_results = []
            
            # 根据输入方式进行搜索
            if text_query:
                with st.spinner('正在搜索相似图片...'):
                    results = search_similar_images(text_query, top_k=top_k)
                    if results:
                        st.session_state.search_results = results
                    else:
                        st.error("未找到相关图片")
            
            elif uploaded_file is not None:
                # 保存上传的图片到临时文件
                temp_path = os.path.join("temp", "uploaded_image.jpg")
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                # 保存上传的图片
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner('正在搜索相似图片...'):
                    results = search_similar_images_by_image(temp_path, top_k=top_k)
                    if results:
                        st.session_state.search_results = results
                    else:
                        st.error("未找到相关图片")
            
            else:
                st.warning("请输入文本描述或上传图片")
    
    with col2:
        st.header("搜索结果")
        
        # 显示搜索结果
        if 'search_results' in st.session_state and st.session_state.search_results:
            # 计算每行显示的图片数
            num_cols = 3  # 每行显示3张图片
            
            # 计算需要的行数
            num_results = len(st.session_state.search_results)
            num_rows = math.ceil(num_results / num_cols)
            
            # 创建瀑布流布局
            for row in range(num_rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    idx = row * num_cols + col_idx
                    if idx < num_results:
                        img_path, similarity = st.session_state.search_results[idx]
                        with cols[col_idx]:
                            try:
                                # 显示图片
                                st.image(img_path, use_container_width=True)
                                # 显示相似度
                                st.caption(f"相似度: {similarity:.4f}")
                                # 显示图片路径（可选）
                                st.caption(f"路径: {os.path.basename(img_path)}")
                            except Exception as e:
                                st.error(f"无法加载图片: {e}")
        else:
            st.info("请输入搜索条件并点击搜索按钮")

if not client.has_collection(COLLECTION_NAME):
    setup_milvus()
    index_images()
