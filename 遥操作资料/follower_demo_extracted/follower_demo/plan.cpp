
#include<aris.hpp>
#include"plan.hpp"


using namespace std;



#include <iostream>
#include <aris/core/core.hpp>
#include <aris/plan/plan.hpp>
#include <aris/robot/rokae.hpp>
#include <functional>
#include <random>

struct Client::Imp
{
	std::unique_ptr<aris::core::Socket> socket_{ new aris::core::Socket };

};
Client::Client(const std::string& name, const std::string& ip, const std::string& port, const aris::core::Socket::Type type) : imp_(new Imp)
{
	
	socket().setPort(port);
	socket().setRemoteIP(ip);
	socket().setConnectType(type);
}
Client::~Client() = default;
auto Client::socket()->aris::core::Socket&
{
	return *imp_->socket_;
}
auto Client::connect()->void { 

	try
	{
		socket().connect();
	}
	catch (const std::exception&e)
	{
		std::cout << e.what() << std::endl;
	}
 }
