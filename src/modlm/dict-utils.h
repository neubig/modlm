#pragma once

#include <memory>
#include <modlm/sentence.h>

namespace dynet { class Dict; }

namespace modlm {

typedef std::shared_ptr<dynet::Dict> DictPtr;
Sentence ParseSentence(const std::string & str, DictPtr dict, bool sent_end);
std::string PrintSentence(const Sentence & sent, DictPtr dict);

}
