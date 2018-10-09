#pragma once
#include <memory/TextLogger.h>
#define tlog(level, fstring, ...) \
  if(tlogger_) \
    tlogger_->logFromLocalization(level, fstring, ##__VA_ARGS__)
