// di/NetworkModule.kt
package com.example.tsukumogami.di

import com.example.tsukumogami.network.GameSocketClient
import com.example.tsukumogami.network.GameSocketClientImpl
import com.google.gson.Gson
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * ネットワーク関連のインスタンス（SocketClientやGson）を
 * Hilt（DIライブラリ）に提供するためのモジュール。
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class NetworkModule {
    @Binds
    @Singleton
    abstract fun bindGameSocketClient(
        gameSocketClientImpl: GameSocketClientImpl
    ): GameSocketClient
}

@Module
@InstallIn(SingletonComponent::class)
object ProviderModule {
    @Provides
    @Singleton
    fun provideGson(): Gson {
        return Gson()
    }
}
