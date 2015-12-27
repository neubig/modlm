#pragma once

#include <memory>
#include <modlm/sentence.h>

namespace cnn { class Dict; }

namespace modlm {

typedef std::shared_ptr<cnn::Dict> DictPtr;
Sentence ParseSentence(const std::string & str, DictPtr dict, bool sent_end);
std::string PrintSentence(const Sentence & sent, DictPtr dict);

}
