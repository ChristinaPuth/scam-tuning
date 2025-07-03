import pandas as pd
import os

# 路径设置
input_path = "src/data/unified_phishing_email_dataset_annotated.csv"
output_path = "src/data/processed_phishing_dataset.csv"

# 读取CSV文件
df = pd.read_csv(input_path)

# 合并 original_subject 和 original_body 为 original_content
df["original_content"] = df["original_subject"].fillna("") + "\n\n" + df["original_body"].fillna("")

# 只保留所需三列
df_filtered = df[["original_content", "class", "explanation"]]

# 保存到指定路径
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_filtered.to_csv(output_path, index=False)

# 打印信息
print(f"✅ 处理完成，已保存到：{output_path}")
print(f"共处理样本数：{len(df_filtered)}")
