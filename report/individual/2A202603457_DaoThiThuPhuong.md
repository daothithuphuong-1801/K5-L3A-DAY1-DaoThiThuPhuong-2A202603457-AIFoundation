# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính


                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.804      0.892        0.947   
2. BoW + MLP (deep learning)         0.988   0.880      0.931        0.974   
3a. Encoder (MiniLM) + kNN           0.869   0.935      0.901        0.976   
3b. Encoder (MiniLM) + LogReg        0.956   0.935      0.945        0.974   
3c. Encoder (MiniLM) + MLP head      0.977   0.935      0.956        0.974   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.079      15316.531       0.050   
2. BoW + MLP (deep learning)        11.616      34191.408       0.322   
3a. Encoder (MiniLM) + kNN          87.581         31.609      42.180   
3b. Encoder (MiniLM) + LogReg       88.039         38.915      38.472   
3c. Encoder (MiniLM) + MLP head     88.915         39.800      25.188   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.082           0.0  
2. BoW + MLP (deep learning)          0.457           0.0  
3a. Encoder (MiniLM) + kNN           64.066           0.0  
3b. Encoder (MiniLM) + LogReg        60.125           0.0  
3c. Encoder (MiniLM) + MLP head      37.814           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.
## 3. Dự đoán và thực tế

### 1. Dự đoán về độ trễ và đường Pareto (tradeoff)

- **Tôi dự đoán:** BoW + Logistic Regression, Embedding + kNN và Embedding + Logistic Regression có khả năng nằm trên đường Pareto và đáp ứng yêu cầu độ trễ dưới 50 ms.
- **Kết quả thực tế:** BoW + LogReg có p95 = 0,082 ms, đáp ứng yêu cầu. Tuy nhiên, Embedding + kNN có p95 = 64,07 ms và Embedding + LogReg có p95 = 60,13 ms, đều không đạt SLA. Đáng chú ý, Embedding + MLP đạt p95 = 37,81 ms và F1 = 0,956, đáp ứng cả yêu cầu độ trễ lẫn chất lượng.
- **Vì sao lệch:** Tôi đã đánh giá thấp thời gian cần thiết để mã hóa văn bản bằng MiniLM. Kết quả cho thấy không thể chỉ dựa vào F1 để dự đoán khả năng đáp ứng SLA mà cần đo trực tiếp độ trễ của toàn bộ quá trình xử lý.

### 2. Dự đoán về Learning Rate (ml_lr)

- **Tôi dự đoán:** Với learning rate = 0,001, loss sẽ giảm rất chậm. Với learning rate = 50, loss có thể dao động mạnh hoặc không hội tụ do bước cập nhật trọng số quá lớn.
- **Kết quả thực tế:** Learning rate = 0,001 đúng là khiến loss giảm chậm. Tuy nhiên, với learning rate = 50, dù loss ban đầu tăng mạnh nhưng sau đó giảm nhanh và đạt khoảng 0,005.
- **Vì sao lệch:** Tôi đã cho rằng learning rate quá lớn chắc chắn khiến mô hình không hội tụ. Thực tế, kết quả còn phụ thuộc vào dữ liệu và quá trình tối ưu. Learning rate lớn có thể gây mất ổn định ban đầu nhưng không nhất thiết khiến mô hình thất bại.

### 3. Dự đoán về F1 của Logistic Regression (ml_f1)

- **Tôi dự đoán:** F1 của mô hình khoảng 0,85, đồng thời Precision có thể cao hơn Recall vì cần hạn chế phân loại nhầm tin nhắn ham thành spam.
- **Kết quả thực tế:** BoW + Logistic Regression đạt F1 = 0,892, Precision = 1,000 và Recall = 0,804.
- **Vì sao lệch:** Tôi đã dự đoán F1 thấp hơn thực tế khoảng 0,042. Tuy nhiên, nhận định Precision cao hơn Recall là chính xác. Mô hình không phân loại nhầm tin ham nào thành spam trên tập test, nhưng vẫn bỏ sót một số tin nhắn spam.

### Kết luận

Qua các thử nghiệm, tôi nhận ra dự đoán lý thuyết không phải lúc nào cũng phản ánh chính xác kết quả thực tế. Cần kiểm chứng bằng số liệu thực nghiệm và đánh giá đồng thời chất lượng phân loại, tốc độ xử lý và độ trễ trước khi lựa chọn mô hình triển khai.

## 4. Phân tích 50 case khó


Dựa trên kết quả đã ghi nhận, Encoder + MLP head (mô hình 3c) có 17/50 case dự đoán sai, tương đương accuracy 66%. Đây là kết quả đáng chú ý vì mô hình này có F1 cao nhất trên tập test thông thường (0,956), nhưng vẫn gặp khó khăn với dữ liệu được thiết kế để thử thách khả năng phân loại.

Nhóm case gây khó khăn:
- Natural_spam: Mô hình sai ở các case #12–16, #18 và #19. Tin nhắn spam có cách diễn đạt tự nhiên, ít từ khóa quảng cáo quen thuộc nên khó nhận diện.
- Manipulation: Mô hình sai ở các case #21–26. Nội dung thường sử dụng cách diễn đạt tạo áp lực, thúc giục hoặc yêu cầu người nhận thực hiện hành động, khiến mô hình dễ nhầm với tin nhắn thông thường.
- Ngoài ra, mô hình còn sai ở nhóm obfuscation, transactional_ham và talks_about_spam.
Hai case cụ thể:

1. Case #12 – Natural_spam: Đây là dạng tin nhắn spam được diễn đạt tự nhiên, không sử dụng rõ ràng những từ khóa spam phổ biến. Mô hình có thể nhận diện sai vì nội dung và ngữ nghĩa bề mặt khá giống một tin nhắn hợp lệ.

2. Case #21 – Manipulation: Đây là dạng tin nhắn có yếu tố thao túng người nhận. Mô hình có thể chưa nhận diện tốt ý đồ lừa đảo ẩn sau cách diễn đạt, đặc biệt khi nội dung không chứa các từ khóa spam quen thuộc.

Kết luận: Kết quả cho thấy F1 cao trên tập test thông thường không đồng nghĩa với khả năng xử lý tốt các tình huống khó. Encoder + MLP có hiệu quả tốt trên dữ liệu thông thường nhưng vẫn có thể mắc lỗi với spam diễn đạt tự nhiên và tin nhắn mang tính thao túng. Vì vậy, cần bổ sung dữ liệu huấn luyện đa dạng và đánh giá riêng theo từng nhóm tình huống.
## 5. Thử thách cá nhân

Đề #2: ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.
- Cách làm
vocab_sizes = [100, 5000]
results = []

for max_size in vocab_sizes:
    # 1. Xây dựng vocab chỉ từ dữ liệu train
    vocab = build_vocab(texts_tr, max_size=max_size)

    # 2. Vector hóa dữ liệu
    X_train = vectorize(texts_tr, vocab)
    X_test = vectorize(texts_te, vocab)

    # 3. Huấn luyện Logistic Regression
    w, b, losses = train_logreg(X_train, y_tr)

    # 4. Dự đoán trên tập test
    y_prob = predict_proba(X_test, w, b)
    y_pred = (y_prob >= 0.5).astype(int)

    # 5. Tính metrics
    scores = metrics(y_te, y_pred)

    results.append({
        "Vocab size": max_size,
        "Actual vocab": len(vocab),
        "F1": scores["f1"],
        "Precision": scores["precision"],
        "Recall": scores["recall"]
    })

# Hiển thị kết quả
pd.DataFrame(results)

- Kết quả: 
	Vocab size	Actual vocab	F1	Precision	Recall
0	100	100	0.834356	0.957746	0.739130
1	5000	5000	0.891566	1.000000	0.804348
 - Giải thích: Khi giảm vocab xuống 100 từ, mô hình có ít đặc trưng đầu vào hơn, dễ bỏ sót các từ khóa quan trọng giúp nhận diện tin nhắn spam nên F1 có thể giảm. Khi tăng vocab lên 5000 từ, mô hình có thêm thông tin để phân biệt spam và ham, từ đó có thể cải thiện F1. Tuy nhiên, vocab quá lớn cũng có thể chứa nhiều từ hiếm hoặc không hữu ích, làm tăng độ phức tạp và nguy cơ overfitting. Vì vậy, cần lựa chọn kích thước vocab phù hợp dựa trên kết quả F1 thực tế.

## 6. Phương án cho sản phẩm thật

Tôi chọn Embedding (MiniLM) + MLP head (mô hình 3c) cho hệ thống lọc 10 triệu SMS/ngày vì mô hình này đáp ứng yêu cầu p95 dưới 50 ms/tin, đồng thời đạt F1 cao nhất trong các phương án đã thử nghiệm. Cụ thể, mô hình đạt F1 = 0,956, Precision = 0,977, Recall = 0,935 và p95 = 37,81 ms, thấp hơn giới hạn 50 ms. BoW + Logistic Regression có tốc độ rất nhanh với p95 = 0,082 ms nhưng F1 chỉ đạt 0,892. Trong khi đó, Embedding + LogReg đạt F1 = 0,945 nhưng p95 = 60,13 ms nên không đáp ứng SLA. Tôi chấp nhận đánh đổi thời gian huấn luyện cao hơn (88,92 giây) và chi phí tài nguyên tính toán cho bước tạo embedding để có chất lượng phân loại tốt hơn. Tuy nhiên, trước khi triển khai thực tế cần kiểm tra lại p95 trên hạ tầng production với tải tương đương 116 tin/giây, vì kết quả benchmark hiện tại chưa đảm bảo hiệu năng khi vận hành ở quy mô lớn.

## 7. AI log và điều đã học

## Nhật ký sử dụng AI

### Lần 1: Logistic Regression
- **Tôi hỏi:** Cách xây dựng mô hình Logistic Regression để phân loại SMS spam.
- **AI trả lời:** Hướng dẫn vector hóa văn bản, tính sigmoid, huấn luyện và đánh giá mô hình.
- **Tôi sửa:** Kiểm tra code, hoàn thiện các phần TODO và chạy thử để xác nhận kết quả.

### Lần 2: Learning Rate
- **Tôi hỏi:** Learning rate ảnh hưởng thế nào đến quá trình huấn luyện?
- **AI trả lời:** Learning rate nhỏ khiến loss giảm chậm, learning rate lớn có thể gây mất ổn định.
- **Tôi sửa:** Đối chiếu với biểu đồ loss thực tế và điều chỉnh nhận xét theo kết quả quan sát.

### Lần 3: MLP và ReLU
- **Tôi hỏi:** So sánh Logistic Regression với MLP khi có và không có ReLU.
- **AI trả lời:** Giải thích vai trò của hàm kích hoạt phi tuyến và nguy cơ overfitting.
- **Tôi sửa:** Đối chiếu F1 và validation loss để đánh giá hiệu quả của từng mô hình.

### Lần 4: Embedding và kNN
- **Tôi hỏi:** Embedding, cosine similarity và kNN hoạt động như thế nào?
- **AI trả lời:** Hướng dẫn biểu diễn ngữ nghĩa văn bản và tìm các tin nhắn tương tự.
- **Tôi sửa:** Kiểm tra kết quả trên các tin nhắn khó và so sánh với Bag-of-Words.

### Lần 5: LLM
- **Tôi hỏi:** Cách xây dựng prompt và sử dụng LLM để phân loại SMS.
- **AI trả lời:** Đề xuất prompt, cách xử lý nhãn và phương pháp Few-shot.
- **Tôi sửa:** Kiểm tra đầu ra, điều chỉnh cách phân tích nhãn và so sánh chất lượng với chi phí API.

### Lần 6: Benchmark và lựa chọn mô hình
- **Tôi hỏi:** Phân tích kết quả benchmark và lựa chọn mô hình đáp ứng yêu cầu p95 dưới 50 ms.
- **AI trả lời:** So sánh F1, tốc độ xử lý, độ trễ p95 và chi phí của các mô hình.
- **Tôi sửa:** Đối chiếu số liệu thực nghiệm và lựa chọn phương án phù hợp với yêu cầu triển khai thực tế.

## Điều tôi hiểu ra hôm nay

Hôm nay tôi hiểu rằng một mô hình có F1 cao chưa chắc là lựa chọn tốt nhất khi triển khai thực tế, vì còn phải cân nhắc độ trễ, chi phí và khả năng xử lý dữ liệu lớn. Tôi cũng nhận ra AI chỉ là công cụ hỗ trợ phân tích và viết code, không nên mặc định mọi câu trả lời đều chính xác mà cần kiểm chứng bằng thực nghiệm. Đặc biệt, việc kết hợp nhiều mô hình theo kiến trúc Router có thể giúp cân bằng chất lượng phân loại, tốc độ và chi phí tốt hơn so với chỉ sử dụng một mô hình duy nhất.

