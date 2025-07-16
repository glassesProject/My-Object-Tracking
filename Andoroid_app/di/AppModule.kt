// di/AppModule.kt
package com.example.tsukumogami.di

import com.example.tsukumogami.data.datasource.remote.GameRemoteDataSource
import com.example.tsukumogami.data.datasource.remote.GameRemoteDataSourceImpl
import com.example.tsukumogami.data.repository.GameRepository
import com.example.tsukumogami.data.repository.GameRepositoryImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class AppModule {

    @Binds
    @Singleton
    abstract fun bindGameRepository(
        gameRepositoryImpl: GameRepositoryImpl
    ): GameRepository

    @Binds
    @Singleton
    abstract fun bindGameRemoteDataSource(
        gameRemoteDataSourceImpl: GameRemoteDataSourceImpl
    ): GameRemoteDataSource
}
