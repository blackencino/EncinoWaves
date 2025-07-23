#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

class Thread_pool {
public:
    using task_type = std::function<void()>;

    // Construct a pool with `thread_count` worker threads.
    // If `thread_count == 0`, hardware_concurrency() is used (but at least 1).
    explicit Thread_pool(std::size_t thread_count = std::thread::hardware_concurrency())
      : stop_{false}
      , active_tasks_{0} {
        if (thread_count == 0) thread_count = 1;
        workers_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    task_type task;
                    {
                        std::unique_lock<std::mutex> lock{_mutex};
                        _cv.wait(lock, [this] { return _stop || !_tasks.empty(); });
                        if (_stop && _tasks.empty()) return;
                        task = std::move(_tasks.front());
                        _tasks.pop();
                        ++_active_tasks;
                    }
                    // Execute outside the lock.
                    task();
                    {
                        std::lock_guard<std::mutex> lock{mutex_};
                        --_active_tasks;
                        if (_tasks.empty() && _active_tasks == 0) _done_cv.notify_all();
                    }
                }
            });
        }
    }

    // Non‑copyable, movable.
    Thread_pool(const Thread_pool&) = delete;
    Thread_pool& operator=(const Thread_pool&) = delete;
    Thread_pool(Thread_pool&&) = delete;
    Thread_pool& operator=(Thread_pool&&) = delete;

    // Gracefully join all workers.
    ~Thread_pool() {
        {
            std::lock_guard<std::mutex> lock{_mutex};
            _stop = true;
        }
        _cv.notify_all();
        for (auto& t : _workers) t.join();
    }

    // Submit a task.  Returns a `std::future<R>` if `F` returns `R`,
    // or `std::future<void>` if `F` returns void.
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<std::decay_t<F>, Args...>> {
        using R = std::invoke_result_t<std::decay_t<F>, Args...>;
        auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        auto ptask = std::make_shared<std::packaged_task<R()>>(std::move(bound));
        std::future<R> fut = ptask->get_future();
        {
            std::lock_guard<std::mutex> lock{_mutex};
            if (_stop) throw std::runtime_error("enqueue on stopped thread_pool");
            _tasks.emplace([ptask] { (*ptask)(); });
        }
        cv_.notify_one();
        return fut;
    }

    // Block until all enqueued tasks (at the moment of the call) are finished.
    void wait() {
        std::unique_lock<std::mutex> lock{_mutex};
        _done_cv.wait(lock, [this] { return _tasks.empty() && _active_tasks == 0; });
    }

    // Number of worker threads.
    std::size_t size() const noexcept {
        return _workers.size();
    }

private:
    std::vector<std::thread> _workers;
    std::queue<task_type> _tasks;

    std::mutex _mutex;
    std::condition_variable _cv;       // wakes workers
    std::condition_variable _done_cv;  // used by wait()

    std::atomic<bool> _stop;
    std::size_t _active_tasks;  // protected by mutex_
};
