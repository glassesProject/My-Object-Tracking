// ui/MainActivity.kt
package com.example.tsukumogami.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.example.tsukumogami.ui.screens.GameScreen
import com.example.tsukumogami.ui.theme.TsukumogamiTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * アプリケーションが起動する最初のエントリーポイント。
 * Jetpack ComposeのUI（GameScreen）を画面に表示する役割を持つ。
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TsukumogamiTheme {
                GameScreen()
            }
        }
    }
}
