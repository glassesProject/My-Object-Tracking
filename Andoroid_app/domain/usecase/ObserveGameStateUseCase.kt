// domain/usecase/ObserveGameStateUseCase.kt
package com.example.tsukumogami.domain.usecase

import com.example.tsukumogami.data.model.GameUiState
import com.example.tsukumogami.data.repository.GameRepository
import com.example.tsukumogami.util.ImageUtils
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject

/**
 * サーバーからの状態を監視するという、単一のビジネスロジック（ユースケース）をカプセル化する。
 */
class ObserveGameStateUseCase @Inject constructor(
    private val repository: GameRepository
) {
    operator fun invoke(): Flow<GameUiState> {
        return repository.connectToServer().map { serverMessage ->
            // サーバーから受け取ったServerMessageを、UIが直接利用できるGameUiStateに変換する
            when (serverMessage.status) {
                "searching" -> GameUiState.Searching(serverMessage.distanceScore ?: 10.0f)
                "locked_on", "processing" -> GameUiState.Processing
                "result" -> {
                    val image = serverMessage.image?.let { ImageUtils.decodeBase64ToBitmap(it) }
                    if (image != null) {
                        GameUiState.ResultReady(image)
                    } else {
                        GameUiState.Error("画像のデコードに失敗しました。")
                    }
                }
                "error" -> GameUiState.Error(serverMessage.message ?: "不明なサーバーエラー")
                else -> GameUiState.Error("予期しないステータス: ${serverMessage.status}")
            }
        }
    }
}
