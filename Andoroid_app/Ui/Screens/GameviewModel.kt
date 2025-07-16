// ui/screens/GameViewModel.kt
package com.example.tsukumogami.ui.screens

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.tsukumogami.data.model.GameUiState
import com.example.tsukumogami.data.repository.GameRepository
import com.example.tsukumogami.domain.usecase.ObserveGameStateUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * UIの状態(State)を保持し、ビジネスロジック（UseCase）とUIの橋渡しをする。
 */
@HiltViewModel
class GameViewModel @Inject constructor(
    observeGameStateUseCase: ObserveGameStateUseCase,
    private val repository: GameRepository // コマンド送信のために追加
) : ViewModel() {

    // ユースケースを通じてサーバーの状態を監視し、UIに公開する
    val uiState: StateFlow<GameUiState> = observeGameStateUseCase()
        .catch { exception ->
            emit(GameUiState.Error(exception.message ?: "不明なエラーが発生しました。"))
        }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = GameUiState.Initializing
        )

    /**
     * ゲーム開始コマンドをサーバーに送信する
     */
    fun startGame() {
        viewModelScope.launch {
            repository.sendCommand("start")
        }
    }

    /**
     * ゲームリスタートコマンドをサーバーに送信する
     */
    fun restartGame() {
        viewModelScope.launch {
            repository.sendCommand("restart")
        }
    }
}