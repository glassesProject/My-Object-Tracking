// data/model/GameUiState.kt
package com.example.tsukumogami.data.model

import androidx.compose.ui.graphics.ImageBitmap

/**
 * アプリケーションのUIが取りうる全ての状態を定義する。
 * これにより、UIはViewModelから渡されたこの状態に応じて表示を切り替えるだけで良くなる。
 */
sealed interface GameUiState {
    // 初期化中・サーバーへの接続待機状態
    object Initializing : GameUiState

    // つくもがみを探している状態。サーバーから送られてくる距離スコアを保持する。
    data class Searching(val distanceScore: Float) : GameUiState

    // ターゲットを発見し、サーバーが画像生成中の状態
    object Processing : GameUiState

    // サーバーから画像を受け取り、結果を表示する準備ができた状態
    data class ResultReady(val tsukumogamiImage: ImageBitmap) : GameUiState

    // エラーが発生した状態
    data class Error(val message: String) : GameUiState
}