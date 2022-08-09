#### 此前

- 請整理上課教室迅速拍照/錄影建模, 然後根據選擇風格/主題自動產生元宇宙教室模型的系統運作流程與需要搭配的技術方法:
  
  - 請繪製系統的區塊圖與流程圖, 然後在區塊圖的一份副本中, 加入各個區塊規畫如何實作的細節:
    
    - 目前使採用 offline 的應用情境:
      
      - 影像資料不夠時, 會需要重新拍攝
      
      - 補充缺失模型的方法也尚未確定 
    
    - 點雲的生成與物件辨識的流程目前調整如下:
      
      - 生成 RGBD 圖片的程式庫:
        
        - https://github.com/NVlabs/neuralrgbd
          
          - 從單目視頻流產生 RGBD 圖片
          
          - Python 有程式庫可以直接讀取 video 裡的 frame (取 frame 時可以跳 frame)
          
          - 已用原作者提供的圖片進行測試成功, 後續將改用自己的圖片/影片進行測試
          
          - 已經可以從 RGB 圖片生成深度圖及 RGBD 圖片 -> 提供給鴻均
          
          - 承霆已經進行過 RGB -> RGBD, 並參考 MVC 視差與深度關係的論文 
            
            -> 確認用單目相機生成 RGBD 的方法並不實際, 考慮採購深度相機 (報帳問題採購前請記得找老師討論)
      
      - 點雲生成的方法, 目前考慮使用的方法為產生 RGBD 影片後再生成點雲 (請先使用 RGBD 的資料集來生成點雲):
        
        - https://github.com/strawlab/python-pcl
        
        - https://github.com/zju3dv/NeuralRecon (需透過 volumn 轉成 point):
          
          - 目前已經可以正常執行, 從影片到生成 3D 模型, 但中間不會生成點雲 (只會生成 3D polygon, 精確度可能不夠)
        
        - 目前改用 Open3D 的點雲生成功能, 測試狀況良好:
          
          - 透過單張圖片產生點雲 + 透過點雲特徵進行配準:
            
            - 目前看測試結果效果不佳 -> 在小型場景內或許可以做到平滑化:
              
              - 小型場景僅需要圖片 20 ~ 30 張, 教室需要 2000 ~ 3000 張
              
              - 目前少量圖片配準誤差不大 -> 看能不能僅使用 EGR 方法, 不會切換到 ICT
          
          - 如果是透過相機矩陣生成點雲, 則效果良好:
            
            - 但除了 RGBD 之外, 還需要 IMU 的相關資訊 -> 若是要使用深度相機, 採購深度相機時需要考慮
          
          - 使用 open3D + 承霆提供的 RGBD 檔案, 目前生成的點雲還有問題:
            
            - 目前看起來問題出在 RGBD 上, 主要是相機的定位可能有問題, 要整合手機 IMU 讀到的資訊, 匯入到產生 RGBD 的程式之介面尚不清楚
            
            - 手機 IMU 數據包括加速度計 (3 軸)、陀螺儀 (3 軸)、磁力計 (3 軸) 共 9 項數據 (目前測試無法使用)
            
            - 實驗測試 open3D 只要給定正確的輸入, 產生的點雲就不會有問題
          
          - Flutter 切換攝影機造成拍攝不順的問題: 建議先使用原始的 480P 模式拍攝, 如果解析度不夠再做調整
        
        - 可能還需要搜尋新的方法生成點雲
      
      - 或是考慮使用 Unity AR Foundation 來生成點雲:
        
        - 目前確定此方法不具可行性, 因為無法從 Unity AR Foundation 內讀取點雲中每個點的座標資訊
      
      - 從點雲進行物件辨識參考的方法 (請開始測試這個部份的實作):
        
        - 目前使用以下套件進行測試:
          
          - https://github.com/cheng052/BRNet
          
          - 目前已經可以框出點雲中的傢俱, 但透過鴻鈞提供的點雲, 辨識的效果略差 (可能跟點雲中椅子 [休閒躺椅] 的型態比較不常見有關)
    
    - 會需要使用神經網路的區塊 (例如: No.2000 及 No.3000):
      
      - 請去確認是否有足夠的訓練資料:
        
        - 確認有 
      
      - 未來考慮分析在訓練資料中加入東海教室的照片, 是不是會有額外的效益
    
    - 請去租用東海 AI 雲進行模型訓練:
      
      - 已經租用了東海 AI 雲上的一台 VM (安裝 Ubuntu 18.4 LTS)
      
      - 目前正在測試能在上面安裝那些程式庫, 而不會造成無法開機的狀況:
        
        - 以下為 VM 內不能自行修改的部份: GPU driver, 桌面啟動相關套件 (例如 GNOME, VMWare 相關套件, Docker) 
      
      - 請完成東海 AI 雲中兩部 VM 的執行環境設定: 
        
        - 已完成
    
    - 請開始分析實作時所需使用的手機及電腦軟硬體平台

- 請整理實作上課教室迅速拍照建模的應用, 需要搭配的軟體 (例如: 景深估算軟體, 點雲產生軟體...等)
  
  - 已完成



#### 其他

##### 链接

[MVR3DR-MonocularVRSpace3DReconstruction/2000-3D_Reconstruction (github.com)](https://github.com/MVR3DR-MonocularVRSpace3DReconstruction/2000-3D_Reconstruction)

[MVR3DR-MonocularVRSpace3DReconstruction/Depth2PointCloud (github.com)](https://github.com/MVR3DR-MonocularVRSpace3DReconstruction/Depth2PointCloud)



##### Minkowski Engine

```ubuntu
# Install MinkowskiEngine
sudo apt install libopenblas-dev g++-7
pip install torch
export CXX=g++-7; pip install -U MinkowskiEngine --install-option="--blas=openblas" -v

# Download and setup DeepGlobalRegistration
git clone https://github.com/chrischoy/DeepGlobalRegistration.git
cd DeepGlobalRegistration
pip install -r requirements.txt
```



借用DGR的安装教程， 但是记得minkowski 的安装需要特定版本，之后过来补充



#### 2022.08.09

现在的专题开发到点云融合的部分，由于多个点云合并会出现重复的平面

现在正在尝试将点云整体平滑化



目前的尝试方案是将点云框进一个box里

并将box分割成多个小区块，再对每个区块做随机的点云采样

```python
def slice_grid_pcds(pcd0, pcd1, box, step):
    box_points = np.asarray(box.get_box_points())
    start_point = [min(box_points[:,0]), min(box_points[:,1]),min(box_points[:,2]) ]
    # print(start_point)
    extent = box.get_extent()
    grid_list = []
    px, py, pz = start_point
    for x in range(1, math.floor((extent[0] * 2 )/step) - 1):
        for y in range(1, math.floor((extent[1] * 2 )/step) - 1):
            for z in range(1, math.floor((extent[1] * 2 )/step) - 1):
                cx = start_point[0] + step * x
                cy = start_point[1] + step * y
                cz = start_point[2] + step * z
                grid_box = o3d.geometry.AxisAlignedBoundingBox([px, py, pz],[cx, cy, cz])
                # print('px py pz cx cy cz\n',[px, py, pz],'\n',[cx, cy, cz],'\n')
                if random.choice([True, False]):
                    grid = pcd0.crop(grid_box)
                else:
                    grid = pcd1.crop(grid_box)
                grid_list.append(grid)
                pz = cz
            pz = start_point[2]
            py = cy
        pz = start_point[2]
        py = start_point[1]
        px = cx
    pcd = concate_pcds(grid_list)
    return pcd

pcd = slice_grid_pcds(pcd0, pcd1, box, 0.3)

# pcd = pcd.crop(box)
o3d.visualization.draw_geometries([pcd,box_0,box_1,box])
```

出现的问题是：

- 采样时间很长
  - step 为 0.1 的情况 需要 五六分钟
- 生成的点云十分混乱
  - <img src="photos\20220809001.png" alt="20220809001" style="zoom:50%;" />
  - <img src="photos\20220809003.jpg" alt="20220809003" style="zoom:50%;" />
  - 且衔接部分出现缺失：
  - <img src="photos\20220809002.jpg" alt="20220809002" style="zoom:50%;" />
  - =

该方案暂不考虑





