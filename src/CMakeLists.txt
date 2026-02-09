ADD_EXECUTABLE(optimizations_ablation optimizations_ablation.cpp)
TARGET_LINK_LIBRARIES(optimizations_ablation fmt::fmt ${SPDLOG_LIBRARIES} Eigen3::Eigen)

ADD_EXECUTABLE(order_ablations order_ablations.cpp)
TARGET_LINK_LIBRARIES(order_ablations fmt::fmt ${SPDLOG_LIBRARIES} Eigen3::Eigen)

ADD_EXECUTABLE(prune_optimizations_ablation prune_optimizations_ablation.cpp)
TARGET_LINK_LIBRARIES(prune_optimizations_ablation fmt::fmt ${SPDLOG_LIBRARIES} Eigen3::Eigen)

ADD_EXECUTABLE(overall_performance overall_performance.cpp)
TARGET_LINK_LIBRARIES(overall_performance fmt::fmt ${SPDLOG_LIBRARIES} Eigen3::Eigen)