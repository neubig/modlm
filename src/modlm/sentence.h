#pragma once

#include <cnn/dict.h>
#include <vector>
#include <cstdint>

namespace modlm {
typedef std::shared_ptr<cnn::Dict> DictPtr;
typedef int32_t WordId;
typedef std::vector<WordId> Sentence;
Sentence ParseSentence(const std::string & str, DictPtr dict, bool sent_end);
std::string PrintSentence(const Sentence & sent, DictPtr dict);
}

namespace std {
  template <> struct hash<modlm::Sentence>
  {
    size_t operator()(const modlm::Sentence & x) const
    { 
      size_t hash = 5381;
      for(auto it = x.begin(); it != x.end(); it++)
          hash = ((hash << 5) + hash) + *it; /* hash * 33 + x[i] */
      return hash;
    }
  };
}
