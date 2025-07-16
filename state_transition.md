graph TD
    A[INITIALIZING] -->|接続成功| B(SEARCHING);
    B -->|発見条件を満たす<br>(サーバ通知)| C(LOCKED_ON);
    C -->|画像処理開始<br>(サーバ通知)| D(PROCESSING);
    D -->|画像受信完了| E(VIEWING_RESULT);
    E -->|再度探索ボタン| B;
    B -->|接続エラー| F(ERROR);
    C -->|接続エラー| F;
    D -->|接続エラー| F;
    F -->|再接続ボタン| A;