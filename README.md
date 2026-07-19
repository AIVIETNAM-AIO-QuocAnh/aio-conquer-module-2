# Ảnh hưởng của Scaling và PCA lên các mô hình học máy cổ điển

Đây là một dự án nghiên cứu nhỏ nhằm khảo sát cách **chuẩn hóa đặc trưng (feature scaling)** và **Phân tích Thành phần Chính (Principal Component Analysis - PCA)** ảnh hưởng đến hiệu năng của các mô hình học máy cổ điển trên bộ dữ liệu **Fashion-MNIST**.

Mục tiêu của dự án không phải là tối đa hóa độ chính xác trên benchmark, mà là so sánh các phương pháp tiền xử lý trong một thiết kế thực nghiệm có kiểm soát và có khả năng tái lập.

## Câu hỏi nghiên cứu

Dự án tập trung trả lời các câu hỏi sau:

1. Feature scaling có làm thay đổi hoặc cải thiện hiệu năng của mô hình hay không?
2. Giảm chiều bằng PCA ảnh hưởng như thế nào đến các nhóm mô hình khác nhau?
3. Những khác biệt quan sát được có nhất quán qua nhiều lần thực nghiệm độc lập hay không?
4. Những khác biệt này có ý nghĩa thống kê hay không?

## Bộ dữ liệu

Dự án sử dụng **Fashion-MNIST**:

- 60.000 ảnh thuộc tập train chính thức
- 10.000 ảnh thuộc tập test chính thức
- 10 lớp tương đối cân bằng
- Ảnh xám kích thước 28 × 28
- Mỗi ảnh tương ứng với 784 đặc trưng sau khi được làm phẳng

### Chia dữ liệu

Tập train và test chính thức được chia riêng thành nhiều **cặp train-test block phân tầng và không chồng lấn**.

Các nguyên tắc chính:

- Mỗi train block được ghép với một test block tương ứng.
- Các train block không chứa mẫu trùng nhau.
- Các test block không chứa mẫu trùng nhau.
- Tất cả mô hình và cấu hình tiền xử lý sử dụng cùng một bộ block.
- Tỷ lệ các lớp được duy trì khi chia dữ liệu.

Việc sử dụng cùng các block cho tất cả cấu hình giúp đảm bảo việc so sánh giữa các pipeline là công bằng.

> Đây không phải là K-Fold Cross-Validation tiêu chuẩn. Các block được sử dụng như những lần thực nghiệm train-test độc lập nhằm đánh giá mức độ nhất quán của ảnh hưởng do preprocessing tạo ra.

Các thông số cụ thể như số block, kích thước block và random seed được quản lý trong file cấu hình của project.

## Các mô hình

Ba mô hình học máy cổ điển được sử dụng:

- **K-Nearest Neighbors (KNN)** — mô hình dựa trên khoảng cách, dự kiến nhạy với scaling và số chiều dữ liệu.
- **Logistic Regression** — mô hình phân loại tuyến tính.
- **Random Forest** — mô hình ensemble dựa trên cây và thường ít nhạy với feature scaling hơn.

Ba mô hình này đại diện cho các nhóm mô hình có đặc tính khác nhau, giúp khảo sát liệu preprocessing có ảnh hưởng khác nhau tùy theo loại mô hình hay không.

## Quy trình thực nghiệm

Với mỗi cặp train-test block và mỗi pipeline:

1. Sử dụng train block để thực hiện hyperparameter tuning bằng cross-validation.
2. Chọn cấu hình có hiệu năng validation tốt nhất.
3. Sử dụng mô hình tốt nhất để đánh giá trên test block tương ứng.
4. Lưu kết quả đánh giá, thời gian chạy, hyperparameters và thông tin của block.
5. Lặp lại quy trình trên tất cả các block.

Các bước preprocessing như `StandardScaler` và `PCA` phải được đặt trong `Pipeline` của scikit-learn để tránh **data leakage**.

Các thông số cụ thể của cross-validation, hyperparameter search và preprocessing được định nghĩa trong file cấu hình.

## Các thí nghiệm

### 1. Ảnh hưởng của Feature Scaling

So sánh hai pipeline:

```text
Dữ liệu gốc -> Mô hình
```

và

```text
StandardScaler -> Mô hình
```

Việc so sánh được thực hiện cho KNN, Logistic Regression và Random Forest trên cùng các cặp train-test block.

Mục tiêu là đánh giá mức độ ảnh hưởng của scaling đối với từng nhóm mô hình.

### 2. Ảnh hưởng của PCA

So sánh:

```text
StandardScaler -> Mô hình
```

với các pipeline sử dụng PCA:

```text
StandardScaler -> PCA -> Mô hình
```

Nhiều mức PCA có thể được sử dụng để khảo sát sự đánh đổi giữa:

- mức độ giảm chiều;
- lượng thông tin được giữ lại;
- hiệu năng phân loại;
- chi phí tính toán.

Pipeline không sử dụng PCA được dùng làm baseline.

Các mức PCA cụ thể được quản lý trong file cấu hình thay vì hard-code trong logic của thí nghiệm.

### 3. Phân tích thống kê

Không cần huấn luyện thêm mô hình cho bước này.

Kết quả của các thí nghiệm trước được sử dụng để thực hiện các phép so sánh theo cặp trên cùng các data block.

Các phép so sánh chính bao gồm:

- Scaling so với không Scaling
- PCA so với không PCA

Ví dụ:

```text
Block 1: Pipeline A vs Pipeline B
Block 2: Pipeline A vs Pipeline B
Block 3: Pipeline A vs Pipeline B
...
```

Mục tiêu là xác định:

- pipeline nào có xu hướng hoạt động tốt hơn;
- sự khác biệt có nhất quán giữa các block hay không;
- sự khác biệt quan sát được có ý nghĩa thống kê hay không.

Phương pháp kiểm định thống kê và mức ý nghĩa được định nghĩa trong kế hoạch phân tích hoặc file cấu hình.

## Chỉ số đánh giá

Dự án tập trung đánh giá các khía cạnh sau:

### Hiệu năng mô hình

Đánh giá khả năng phân loại của từng mô hình và preprocessing pipeline.

### Độ ổn định

Đánh giá mức độ ổn định của kết quả qua nhiều lần thực nghiệm độc lập.

### Chi phí tính toán

Theo dõi các thông tin như:

- thời gian huấn luyện;
- thời gian suy luận;
- mức giảm số chiều sau PCA.

### So sánh thống kê

So sánh kết quả theo cặp giữa các pipeline để đánh giá mức độ nhất quán và ý nghĩa thống kê của sự khác biệt.

Các metric cụ thể được quản lý trong file cấu hình của project.

## Tìm kiếm siêu tham số

Mỗi mô hình sử dụng một không gian tìm kiếm siêu tham số phù hợp.

Các nguyên tắc chính:

- Cùng một mô hình phải sử dụng cùng hyperparameter search space khi so sánh các preprocessing configurations.
- Hyperparameter tuning chỉ được thực hiện trên training data.
- Test data không được sử dụng để chọn mô hình hoặc siêu tham số.
- Không gian tìm kiếm được quản lý trong file cấu hình thay vì hard-code trong source code.

## Kết quả đầu ra

Project dự kiến tạo ra:

- kết quả so sánh Scaling và không Scaling;
- kết quả so sánh PCA và không PCA;
- kết quả kiểm định thống kê;
- so sánh thời gian chạy;
- hyperparameters tốt nhất của từng thí nghiệm;
- biểu đồ so sánh theo cặp;
- boxplot kết quả trên nhiều block;
- phân tích ảnh hưởng của PCA đến hiệu năng và số chiều dữ liệu.

Kết quả của từng block cần được lưu riêng, thay vì chỉ lưu giá trị trung bình cuối cùng, để phục vụ việc phân tích thống kê sau này.

## Vai trò trong nhóm

| Vai trò | Trách nhiệm |
|---|---|
| Tech Leader | Quản lý tiến độ, thiết kế thực nghiệm, điều phối công việc, tích hợp và xử lý các thay đổi trong thiết kế |
| AI Engineer — Data | Chuẩn bị dữ liệu, tạo các data block phân tầng, quản lý seed và indices, kiểm tra tính toàn vẹn của dữ liệu |
| AI Engineer — Pipeline | Xây dựng pipeline thực nghiệm dùng chung, thiết lập hyperparameter tuning, phụ trách thí nghiệm scaling và phân tích thống kê tương ứng |
| AI Engineer — Model | Xây dựng các model definitions và hyperparameter configurations, phụ trách thí nghiệm PCA và phân tích thống kê tương ứng |
| QA / Reviewer | Kiểm tra data split, data leakage, khả năng tái lập, tính nhất quán giữa các thí nghiệm, kết quả và báo cáo cuối |

## Phạm vi dự án

### Nội dung nằm trong phạm vi

- Fashion-MNIST
- Các mô hình học máy cổ điển
- Feature Scaling
- PCA
- Hyperparameter tuning
- Model evaluation
- Phân tích thống kê
- Khả năng tái lập của thí nghiệm

### Nội dung ngoài phạm vi

- Deep Learning hoặc CNN
- Tối ưu để đạt state-of-the-art performance
- Benchmark số lượng lớn các mô hình
- Hyperparameter optimization quy mô lớn
- Bổ sung dataset, model hoặc preprocessing method mới nếu chưa được thống nhất

## Khả năng tái lập

Để đảm bảo việc so sánh công bằng và có thể tái lập:

- sử dụng random seed cố định;
- sử dụng cùng data split cho tất cả thí nghiệm;
- lưu lại các block indices;
- sử dụng cùng hyperparameter search space cho cùng một mô hình;
- test data không được sử dụng trong model selection;
- preprocessing chỉ được fit từ training data;
- kết quả được lưu ở cấp từng block;
- các thông số thực nghiệm được quản lý tập trung trong file cấu hình.

## Cấu hình thực nghiệm

Các thông số có thể thay đổi trong quá trình nghiên cứu được quản lý trong file cấu hình.

Ví dụ:

```text
configs/
    experiment.yaml
```

File cấu hình có thể chứa:

```text
dataset settings
data split settings
random seed

cross-validation settings

model configurations
hyperparameter search spaces

scaling settings
PCA settings

evaluation metrics

result paths
```

Source code nên đọc các thông số này từ config thay vì hard-code trực tiếp trong logic của chương trình.

Cách tổ chức này giúp thay đổi thiết kế thí nghiệm mà không cần chỉnh sửa nhiều source code.

## Trạng thái dự án

Project hiện đang trong giai đoạn xây dựng và thực nghiệm.

Các nội dung sau sẽ được cập nhật sau khi hoàn thành thí nghiệm:

- cấu hình thực nghiệm cuối cùng;
- kết quả;
- biểu đồ;
- phân tích thống kê;
- thảo luận;
- kết luận.