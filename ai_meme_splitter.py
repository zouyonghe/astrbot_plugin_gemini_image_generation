import cv2
import numpy as np
from typing import List, Tuple
import os


class AIMemeSplitter:
    """
    AI辅助表情包切分器
    接收AI识别的行列数，使用meme_splitter_v2.py的算法进行切图
    """
    
    def __init__(self, min_gap=10, edge_threshold=15):
        """
        Args:
            min_gap: 最小间隙宽度（像素）
            edge_threshold: 边缘检测阈值
        """
        self.min_gap = min_gap
        self.edge_threshold = edge_threshold
        self.process_steps = {}  # 存储处理步骤图像
        self.last_row_lines = []
        self.last_col_lines = []
        self.detected_rows = 0
        self.detected_cols = 0
        self.analysis_info = ""  # 分析过程信息

    def dilate_diff(self, img):
        """
        通过膨胀差分提取线条，适合动漫 (来自meme_splitter_v2.py)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 膨胀图像
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # 计算差值
        diff = cv2.absdiff(gray, dilated)
        
        # 反转并增强
        result = 255 - diff
        
        # 再次阈值化以增强黑白对比
        _, result = cv2.threshold(result, 230, 255, cv2.THRESH_BINARY)
        
        return result

    def analyze_grid_auto(self, lineart):
        """
        自动分析网格的行列数
        通过多处采样计算周期距离和FFT分析
        
        Returns:
            (rows, cols, analysis_info): 检测到的行数、列数和分析信息
        """
        h, w = lineart.shape
        edges = 255 - lineart
        
        analysis_log = []
        analysis_log.append("=" * 50)
        analysis_log.append("开始自动网格分析")
        analysis_log.append("=" * 50)
        
        # 1. 投影分析
        row_proj = np.sum(edges, axis=1)
        col_proj = np.sum(edges, axis=0)
        
        # 2. FFT 频谱分析检测周期
        def fft_period_analysis(projection, axis_name):
            """使用FFT检测投影的主要周期"""
            # 去除直流分量
            proj_norm = projection - np.mean(projection)
            
            # 应用汉宁窗减少频谱泄露
            window = np.hanning(len(proj_norm))
            proj_windowed = proj_norm * window
            
            # FFT变换
            fft_result = np.fft.fft(proj_windowed)
            freqs = np.fft.fftfreq(len(proj_windowed))
            
            # 只取正频率部分
            positive_freqs = freqs[:len(freqs)//2]
            magnitude = np.abs(fft_result[:len(fft_result)//2])
            
            # 排除过高和过低的频率（对应的周期太小或太大）
            min_period = 20  # 最小周期（像素）
            max_period = len(projection) // 2  # 最大周期
            
            valid_mask = (positive_freqs > 1.0/max_period) & (positive_freqs < 1.0/min_period)
            valid_freqs = positive_freqs[valid_mask]
            valid_magnitude = magnitude[valid_mask]
            
            if len(valid_magnitude) == 0:
                return None, 0
            
            # 找到最强的几个频率峰
            peak_indices = []
            sorted_indices = np.argsort(valid_magnitude)[::-1]
            
            for idx in sorted_indices[:5]:  # 取前5个峰
                freq = valid_freqs[idx]
                period = 1.0 / freq if freq > 0 else 0
                mag = valid_magnitude[idx]
                if period > min_period and period < max_period:
                    peak_indices.append((period, mag))
            
            if not peak_indices:
                return None, 0
                
            # 选择最强的周期
            best_period, best_mag = peak_indices[0]
            return best_period, best_mag
        
        analysis_log.append(f"\n图像尺寸: {h} x {w}")
        
        # 行分析
        row_period, row_mag = fft_period_analysis(row_proj, "行")
        if row_period:
            estimated_rows = int(round(h / row_period))
            analysis_log.append(f"\nFFT行分析:")
            analysis_log.append(f"  检测到周期: {row_period:.1f} 像素")
            analysis_log.append(f"  频谱强度: {row_mag:.1f}")
            analysis_log.append(f"  估计行数: {estimated_rows}")
        else:
            estimated_rows = 3
            analysis_log.append(f"\nFFT行分析: 未检测到明显周期，默认 {estimated_rows} 行")
        
        # 列分析
        col_period, col_mag = fft_period_analysis(col_proj, "列")
        if col_period:
            estimated_cols = int(round(w / col_period))
            analysis_log.append(f"\nFFT列分析:")
            analysis_log.append(f"  检测到周期: {col_period:.1f} 像素")
            analysis_log.append(f"  频谱强度: {col_mag:.1f}")
            analysis_log.append(f"  估计列数: {estimated_cols}")
        else:
            estimated_cols = 3
            analysis_log.append(f"\nFFT列分析: 未检测到明显周期，默认 {estimated_cols} 列")
        
        # 3. 多采样点验证
        def multi_sample_verification(projection, estimated_n, axis_name):
            """通过多处采样验证估计的数量 - 改进版"""
            length = len(projection)
            estimated_period = length / estimated_n
            
            # 使用更强的平滑以减少噪声
            kernel_size = max(5, int(estimated_period * 0.2))
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = np.convolve(projection, np.ones(kernel_size)/kernel_size, mode='same')
            
            # 归一化
            smoothed_norm = (smoothed - np.min(smoothed)) / (np.max(smoothed) - np.min(smoothed) + 1e-10)
            
            try:
                from scipy import signal
                # 使用更严格的参数找波谷
                # prominence: 波峰的突出程度
                # distance: 最小间隔
                # width: 波峰宽度
                min_distance = int(estimated_period * 0.7)  # 至少70%的预期周期
                
                # 找波谷（反转信号）
                peaks, properties = signal.find_peaks(
                    -smoothed_norm,
                    distance=min_distance,
                    prominence=0.1,  # 至少要有10%的突出度
                    width=3  # 至少3个点宽
                )
                
                detected_n = len(peaks) + 1
                
                analysis_log.append(f"\n{axis_name}多采样验证 (scipy):")
                analysis_log.append(f"  预期周期: {estimated_period:.1f} 像素")
                analysis_log.append(f"  预期数量: {estimated_n}")
                analysis_log.append(f"  检测到 {len(peaks)} 个分割点")
                analysis_log.append(f"  初步验证数量: {detected_n}")
                
                # 决策逻辑：优先信任FFT结果
                if abs(detected_n - estimated_n) <= 1:
                    # 差异在1以内，使用FFT估计
                    final_n = estimated_n
                    analysis_log.append(f"  ✓ 验证一致，使用FFT结果: {final_n}")
                elif abs(detected_n - estimated_n) == 2 and detected_n > estimated_n:
                    # 如果多检测了2个，可能是误检，倾向FFT
                    final_n = estimated_n
                    analysis_log.append(f"  ⚠ 多采样偏高，信任FFT: {final_n}")
                else:
                    # 差异较大时，取平均或中间值
                    final_n = int(round((detected_n + estimated_n) / 2))
                    analysis_log.append(f"  ⚠ 差异较大，取折中: {final_n}")
                
                return final_n
                
            except ImportError:
                # 没有scipy，使用改进的简单峰检测
                analysis_log.append(f"\n{axis_name}验证 (简化版):")
                
                # 使用梯度检测波谷
                gradient = np.gradient(smoothed_norm)
                
                # 找零crossing (从负到正 = 波谷)
                zero_crossings = []
                for i in range(1, len(gradient)):
                    if gradient[i-1] < 0 and gradient[i] >= 0:
                        zero_crossings.append(i)
                
                # 过滤太近的点
                if zero_crossings:
                    min_dist = int(estimated_period * 0.7)
                    filtered = [zero_crossings[0]]
                    for zc in zero_crossings[1:]:
                        if zc - filtered[-1] >= min_dist:
                            filtered.append(zc)
                    detected_n = len(filtered) + 1
                else:
                    detected_n = estimated_n
                
                analysis_log.append(f"  预期周期: {estimated_period:.1f} 像素")
                analysis_log.append(f"  检测到 {len(zero_crossings)} 个候选点")
                analysis_log.append(f"  过滤后: {len(filtered) if zero_crossings else 0} 个分割点")
                
                # 同样的决策逻辑
                if abs(detected_n - estimated_n) <= 1:
                    final_n = estimated_n
                    analysis_log.append(f"  ✓ 使用FFT结果: {final_n}")
                else:
                    final_n = estimated_n  # 简化版优先信任FFT
                    analysis_log.append(f"  ⚠ 信任FFT: {final_n}")
                
                return final_n
        
        final_rows = multi_sample_verification(row_proj, estimated_rows, "行")
        final_cols = multi_sample_verification(col_proj, estimated_cols, "列")
        
        # 范围限制
        final_rows = max(1, min(10, final_rows))
        final_cols = max(1, min(10, final_cols))
        
        analysis_log.append("\n" + "=" * 50)
        analysis_log.append(f"最终结果: {final_rows} 行 x {final_cols} 列")
        analysis_log.append("=" * 50)
        
        self.detected_rows = final_rows
        self.detected_cols = final_cols
        self.analysis_info = "\n".join(analysis_log)
        
        return final_rows, final_cols, self.analysis_info

    def post_process(self, lineart, threshold=50):
        """
        后处理：去除小的连通域（杂线、噪点）(来自meme_splitter_v2.py)
        """
        # 反转为黑底白线以便查找连通域
        binary = 255 - lineart
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 创建新的掩码
        new_mask = np.zeros_like(binary)
        
        for i in range(1, num_labels):  # 跳过背景 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= threshold:
                new_mask[labels == i] = 255
                
        # 反转回白底黑线
        result = 255 - new_mask
        
        return result

    def detect_grid(self, lineart, target_rows: int, target_cols: int):
        """
        智能网格检测，使用AI提供的行列数作为目标，支持网格线智能调整
        
        Args:
            lineart: 线稿图像
            target_rows: 目标行数（AI识别的）
            target_cols: 目标列数（AI识别的）
            
        Returns:
            (h_lines, v_lines): 水平线和垂直线位置列表
        """
        h, w = lineart.shape
        edges = 255 - lineart
        
        # 1. 预处理 - 结构线提取
        k_w = max(3, w // 5)
        if k_w % 2 == 0: k_w += 1
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        h_struct = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        h_struct_proj = np.sum(h_struct, axis=1)
        
        k_h = max(3, h // 5)
        if k_h % 2 == 0: k_h += 1
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        v_struct = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        v_struct_proj = np.sum(v_struct, axis=0)
        
        # 缝隙投影
        kernel_size = 3
        row_proj = np.convolve(np.sum(edges, axis=1), np.ones(kernel_size)/kernel_size, mode='same')
        col_proj = np.convolve(np.sum(edges, axis=0), np.ones(kernel_size)/kernel_size, mode='same')

        def optimize_grid_positions(initial_cuts, proj_values, content_proj, length, axis_name):
            """
            优化网格位置：在保证均匀性的前提下，避免切割内容轮廓
            
            Args:
                initial_cuts: 初始切割位置
                proj_values: 间隙投影值（越小越适合切割）
                content_proj: 内容投影值（越大内容越多）
                length: 轴长度
                axis_name: 轴名称
            """
            if len(initial_cuts) <= 2:  # 只有边界，无需优化
                return initial_cuts
            
            optimized_cuts = [initial_cuts[0]]  # 保持起始边界
            
            for i in range(1, len(initial_cuts) - 1):  # 跳过边界
                current_pos = initial_cuts[i]
                
                # 计算理想位置（基于均匀分布）
                ideal_interval = length / (len(initial_cuts) - 1)
                ideal_pos = int(i * ideal_interval)
                
                # 定义搜索范围（理想位置附近）
                search_radius = min(int(ideal_interval * 0.3), 20)  # 最多偏移30%或20像素
                search_start = max(0, ideal_pos - search_radius)
                search_end = min(length, ideal_pos + search_radius)
                
                best_pos = current_pos
                best_score = float('-inf')
                
                # 在搜索范围内寻找最佳位置
                for test_pos in range(search_start, search_end):
                    # 1. 间隙分数（越小越好，表示内容越少）
                    gap_score = 1.0 / (1.0 + proj_values[test_pos])
                    
                    # 2. 均匀性分数
                    if i == 1:  # 第一个内部切割线
                        prev_interval = test_pos - optimized_cuts[0]
                        next_interval = ideal_interval  # 假设后续均匀
                    else:
                        prev_interval = test_pos - optimized_cuts[-1]
                        next_interval = ideal_interval
                    
                    # 计算间隔的标准差（越小越均匀）
                    intervals = [prev_interval, next_interval]
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    uniformity_score = 1.0 / (1.0 + std_interval / mean_interval) if mean_interval > 0 else 0
                    
                    # 3. 距离理想位置的惩罚
                    distance_penalty = 1.0 / (1.0 + abs(test_pos - ideal_pos) / search_radius)
                    
                    # 综合评分：均匀性优先，间隙次之，距离最后
                    total_score = (0.6 * uniformity_score + 
                                 0.3 * gap_score + 
                                 0.1 * distance_penalty)
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_pos = test_pos
                
                optimized_cuts.append(best_pos)
                print(f"[{axis_name}] 位置{i}: {current_pos} -> {best_pos} (偏移{best_pos-current_pos})")
            
            optimized_cuts.append(initial_cuts[-1])  # 保持结束边界
            return optimized_cuts

        def solve_axis(gap_proj, struct_proj, length, axis_name, manual_n):
            """使用AI提供的行列数进行网格检测，然后优化位置"""
            # 标准化
            max_gap = np.max(gap_proj)
            norm_gap = gap_proj / max_gap if max_gap > 0 else gap_proj
            
            max_struct = np.max(struct_proj)
            norm_struct = struct_proj / max_struct if max_struct > 0 else struct_proj
            
            # 安全阈值
            sorted_vals = np.sort(norm_gap)
            baseline = np.mean(sorted_vals[:int(length * 0.1) + 1])
            safe_threshold = baseline + 0.25

            best_score = -float('inf')
            best_cuts = [0, length]
            best_mode = "None"
            
            n = manual_n
            if n == 1:
                return [0, length]
                
            step = length / n
            
            # --- 尝试两种模式 ---
            modes = []
            
            # Mode A: Gap (找波谷)
            gap_cuts = [0]
            gap_vals = []
            gap_displacements = []
            valid_gap = True
            
            for k in range(1, n):
                ideal = int(k * step)
                radius = int(step * 0.25)
                start = max(0, ideal - radius)
                end = min(length, ideal + radius)
                
                window = norm_gap[start:end]
                if len(window) == 0: 
                    valid_gap = False
                    break
                
                idx = np.argmin(window)
                pos = start + idx
                val = window[idx]
                
                # 硬性过滤
                if val > safe_threshold * 1.5:
                    valid_gap = False
                    break
                    
                gap_cuts.append(pos)
                gap_vals.append(val)
                gap_displacements.append(abs(pos - ideal))
            
            if valid_gap:
                gap_cuts.append(length)
                modes.append(("Gap", gap_cuts, gap_vals, gap_displacements))

            # Mode B: Structure (找波峰)
            if max_struct > 0:
                struct_cuts = [0]
                struct_vals = []
                struct_displacements = []
                valid_struct = True
                
                for k in range(1, n):
                    ideal = int(k * step)
                    radius = int(step * 0.2)
                    start = max(0, ideal - radius)
                    end = min(length, ideal + radius)
                    
                    window = norm_struct[start:end]
                    if len(window) == 0: 
                        valid_struct = False
                        break
                    
                    idx = np.argmax(window)
                    pos = start + idx
                    val = window[idx]
                    
                    if val < 0.2:
                        valid_struct = False
                        break
                        
                    struct_cuts.append(pos)
                    struct_vals.append(1.0 - val)
                    struct_displacements.append(abs(pos - ideal))
                
                if valid_struct:
                    struct_cuts.append(length)
                    modes.append(("Struct", struct_cuts, struct_vals, struct_displacements))
            
            # --- 评分并选择最佳模式 ---
            for mode_name, cuts, vals, displacements in modes:
                # 1. 均匀度
                intervals = np.diff(cuts)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                cv = std_interval / mean_interval if mean_interval > 0 else 0
                score_uniformity = max(0, 1.0 - (cv / 0.3))
                
                # 2. 安全度
                avg_val = np.mean(vals)
                score_safety = max(0, 1.0 - avg_val)
                
                # 3. 变动度
                avg_disp = np.mean(displacements)
                max_disp = step * 0.3
                score_displacement = max(0, 1.0 - (avg_disp / max_disp))
                
                # 加权总分
                final_score = (0.5 * score_uniformity + 
                               0.3 * score_safety + 
                               0.2 * score_displacement) + (n * 0.05)
                               
                if final_score > best_score:
                    best_score = final_score
                    best_cuts = cuts
                    best_mode = f"{mode_name}(N={n})"

            print(f"[{axis_name}] 初始方案: {best_mode}, 得分={best_score:.3f}")
            
            # 对选定的切割位置进行智能优化
            if len(best_cuts) > 2:
                print(f"[{axis_name}] 开始位置优化...")
                # 计算内容投影（边缘密度）
                content_proj = gap_proj  # 使用间隙投影的反向作为内容投影
                optimized_cuts = optimize_grid_positions(best_cuts, norm_gap, content_proj, length, axis_name)
                
                # 验证优化后的均匀性
                old_intervals = np.diff(best_cuts)
                new_intervals = np.diff(optimized_cuts)
                old_cv = np.std(old_intervals) / np.mean(old_intervals) if np.mean(old_intervals) > 0 else float('inf')
                new_cv = np.std(new_intervals) / np.mean(new_intervals) if np.mean(new_intervals) > 0 else float('inf')
                
                if new_cv <= old_cv * 1.1:  # 允许轻微的均匀性降低（10%内）
                    print(f"[{axis_name}] 优化成功: CV {old_cv:.3f} -> {new_cv:.3f}")
                    best_cuts = optimized_cuts
                else:
                    print(f"[{axis_name}] 优化被拒绝: CV会从 {old_cv:.3f} 变为 {new_cv:.3f}")
            
            return best_cuts

        h_lines = solve_axis(row_proj, h_struct_proj, h, "水平", target_rows)
        v_lines = solve_axis(col_proj, v_struct_proj, w, "垂直", target_cols)
        
        print(f"最终网格: {len(h_lines)-1}行 x {len(v_lines)-1}列")
        return h_lines, v_lines

    def split(self, image_path: str, output_dir: str, rows: int, cols: int,
              debug: bool = False) -> List[str]:
        """
        使用AI识别的行列数切分图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            rows: AI识别的行数
            cols: AI识别的列数
            debug: 是否显示调试信息
            
        Returns:
            保存的文件路径列表
        """
        # 读取图像
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        self.process_steps['1_original'] = img_original.copy()
        
        # 1. 线稿提取 (Dilate Diff)
        if debug:
            print("正在提取线稿...")
        img_lineart = self.dilate_diff(img_original)
        self.process_steps['2_lineart'] = img_lineart.copy()
        
        # 2. 后处理 (去杂线)
        if debug:
            print("正在去除杂线...")
        img_clean = self.post_process(img_lineart, threshold=50)
        self.process_steps['3_clean'] = img_clean.copy()
        
        # 3. 网格检测（使用AI提供的行列数）
        if debug:
            print(f"正在按 {rows}行 x {cols}列 检测网格...")
        h_lines, v_lines = self.detect_grid(img_clean, rows, cols)
        
        # 保存网格线信息
        self.last_row_lines = h_lines
        self.last_col_lines = v_lines
        
        # 4. 生成切分区域
        boxes = []
        centers = []
        
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            for i in range(len(h_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i+1]
                cy = (y1 + y2) // 2
                for j in range(len(v_lines) - 1):
                    x1, x2 = v_lines[j], v_lines[j+1]
                    cx = (x1 + x2) // 2
                    w = x2 - x1
                    h = y2 - y1
                    centers.append((cx, cy, w, h))
        
        # 5. 生成边界框
        for cx, cy, w, h in centers:
            x1 = cx - w//2
            y1 = cy - h//2
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_original.shape[1], x1 + w)
            y2 = min(img_original.shape[0], y1 + h)
            
            # 验证区域有足够内容
            if x2 - x1 > 20 and y2 - y1 > 20:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        if not boxes:
            print("未检测到有效的表情包区域")
            return []
        
        # 6. 保存切分结果
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for idx, (x, y, w, h) in enumerate(boxes, 1):
            crop = img_original[y:y+h, x:x+w]
            filename = f"meme_{idx:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, crop)
            saved_files.append(filepath)
        
        if debug:
            print(f"成功保存 {len(saved_files)} 个表情包到 {output_dir}")
        
        return saved_files
    
    def draw_grid_lines(self, image: np.ndarray, edges: np.ndarray, 
                       row_lines: List[int], col_lines: List[int]) -> np.ndarray:
        """
        绘制完整的网格线（一行一列直线）
        """
        h, w = image.shape[:2]
        
        # 转换为RGB用于绘制
        if len(image.shape) == 3 and image.shape[2] == 4:
            display_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            display_img = image.copy()
        
        # 绘制水平线（行）
        for y in row_lines:
            if 0 <= y < h:
                cv2.line(display_img, (0, y), (w, y), (0, 255, 0), 2)
        
        # 绘制垂直线（列）
        for x in col_lines:
            if 0 <= x < w:
                cv2.line(display_img, (x, 0), (x, h), (0, 255, 0), 2)
        
        return display_img


# 使用示例
if __name__ == "__main__":
    import asyncio
    import openai
    
    async def test():
        # 初始化客户端
        openai_client = openai.AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="dummy"
        )
        
        # 创建切分器
        splitter = AIMemeSplitter(min_gap=10, edge_threshold=15)
        
        # 切分图片
        image_path = "test.png"
        output_dir = "output"
        
        saved_files = await splitter.split(openai_client, image_path, output_dir, debug=True)
        print(f"切分完成，共生成 {len(saved_files)} 个文件")
    
    asyncio.run(test())