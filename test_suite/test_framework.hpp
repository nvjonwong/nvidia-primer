/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

// Simple test framework
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

class TestRunner {
    std::vector<TestResult> results_;
    
public:
    void run_test(const std::string& name, std::function<void()> test_func) {
        std::cout << "Running: " << name << "... " << std::flush;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool passed = false;
        std::string message;
        
        try {
            test_func();
            passed = true;
            message = "OK";
            std::cout << "PASSED" << std::endl;
        } catch (const std::exception& e) {
            passed = false;
            message = std::string("Exception: ") + e.what();
            std::cout << "FAILED: " << message << std::endl;
        } catch (...) {
            passed = false;
            message = "Unknown exception";
            std::cout << "FAILED: " << message << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        results_.push_back({name, passed, message, duration});
    }
    
    void print_summary() {
        std::cout << "\n" << std::string(70, '=\) << "\n";
        std::cout << "TEST SUMMARY\n";
        std::cout << std::string(70, '=\) << "\n";
        
        int passed = 0;
        int failed = 0;
        double total_time = 0.0;
        
        for (const auto& result : results_) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
                std::cout << "FAILED: " << result.name << " - " << result.message << "\n";
            }
            total_time += result.duration_ms;
        }
        
        std::cout << "\nTotal: " << results_.size() << " tests\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Total time: " << total_time << " ms\n";
        std::cout << std::string(70, '=\) << "\n";
    }
    
    bool all_passed() const {
        for (const auto& result : results_) {
            if (!result.passed) return false;
        }
        return true;
    }
};
