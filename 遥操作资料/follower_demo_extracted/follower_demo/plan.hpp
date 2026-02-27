#ifndef PLAN_HPP
#define PLAN_HPP

#include<aris.hpp>

#include"plan.hpp"

using namespace std;
using namespace aris::plan;
using namespace aris::dynamic;
using namespace aris::core;



class Client
{
public:
	Client(const std::string& name = "Client", const std::string& ip = "192.168.31.166",const std::string& port = "1234", const aris::core::Socket::Type type = aris::core::Socket::Type::WEB);
	~Client();

	auto socket()->aris::core::Socket&;
	auto connect()->void;

	ARIS_DECLARE_BIG_FOUR(Client);
private:
	struct Imp;
	std::unique_ptr<Imp> imp_;

};






#endif // !PLAN_HPP

