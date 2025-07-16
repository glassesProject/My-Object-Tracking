// ui/screens/GameScreen.kt
package com.example.tsukumogami.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

/**
 * ゲームのメイン画面。
 * ViewModelから受け取った状態(uiState)に応じて、表示するUIを切り替える。
 */
@Composable
fun GameScreen(viewModel: GameViewModel = hiltViewModel()) {
    val uiState by viewModel.uiState.collectAsState()

    Surface(modifier = Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            when (val state = uiState) {
                is GameUiState.Initializing -> InitializingView(onStartClick = { viewModel.startGame() })
                is GameUiState.Searching -> SearchingView(distanceScore = state.distanceScore)
                is GameUiState.Processing -> ProcessingView()
                is GameUiState.ResultReady -> ResultView(
                    image = state.tsukumogamiImage,
                    onRestartClick = { viewModel.restartGame() }
                )
                is GameUiState.Error -> ErrorView(message = state.message)
            }
        }
    }
}

@Composable
fun InitializingView(onStartClick: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("つくもがみを探す", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(24.dp))
        Button(onClick = onStartClick) {
            Text("ゲーム開始")
        }
        Spacer(Modifier.height(8.dp))
        Text("PCサーバーに接続します...", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
fun SearchingView(distanceScore: Float) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("探索中...", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(16.dp))
        val progressColor = when {
            distanceScore < 2.5f -> Color.Red
            distanceScore < 4.5f -> Color(0xFFE6E600) // Yellow
            else -> Color.Gray
        }
        CircularProgressIndicator(
            modifier = Modifier.size(80.dp),
            color = progressColor,
            strokeWidth = 8.dp
        )
        Spacer(Modifier.height(16.dp))
        Text("ターゲットとの距離: ${"%.1f".format(distanceScore)}")
    }
}

@Composable
fun ProcessingView() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("つくもがみ発見！", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(16.dp))
        CircularProgressIndicator()
        Spacer(Modifier.height(8.dp))
        Text("写真を生成しています…")
    }
}

@Composable
fun ResultView(image: ImageBitmap, onRestartClick: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("撮影完了！", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(16.dp))
        Image(
            bitmap = image,
            contentDescription = "つくもがみの写真",
            modifier = Modifier.size(300.dp)
        )
        Spacer(Modifier.height(24.dp))
        Button(onClick = onRestartClick) {
            Text("もう一度探す")
        }
    }
}

@Composable
fun ErrorView(message: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("エラー", style = MaterialTheme.typography.headlineMedium, color = Color.Red)
        Spacer(Modifier.height(8.dp))
        Text(message, style = MaterialTheme.typography.bodyLarge)
    }
}