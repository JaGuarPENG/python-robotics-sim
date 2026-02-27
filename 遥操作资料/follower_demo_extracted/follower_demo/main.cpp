

#include<chrono>
#include<ctime>
#include<thread>
#include<iostream>
#include<aris.hpp>
#include<fstream>
#include<atomic>
#include <iostream>
#include <aris/core/core.hpp>
#include <aris/plan/plan.hpp>
#include <aris/robot/rokae.hpp>
#include <random>
#include <numeric>
#include"plan.hpp"
#include"json.hpp"

using namespace std;


int main(int argc, char *argv[])
{

	Client client("client","192.168.192.149", "9998", aris::core::Socket::Type::UDP);

	
	
	auto onLoseConnection = [&](aris::core::Socket* socket)->int {

		std::cout << "client lose connect, reconnect" << std::endl;
		socket->connect();
		return 0;
	};

	auto onReceivedMsg = [](aris::core::Socket* socket, aris::core::Msg& msg)->int {

		std::cout << msg.size() << std::endl;
		std::cout << msg.data() << std::endl;
		std::cout << "client onReceivedMsg" << std::endl;
		//std::cout << std::string_view(msg.data(), msg.size()) << std::endl;
		return 0;
	};
	aris::core::Msg msg;
	// 关节
	//msg.copy("jp --motor_pos=[0,-5,90,10,-100,0]");
	// 笛卡尔
	nlohmann::json j;
	std::vector<std::vector<double>> pes,pqs;
	std::vector<double> pe{ 0.10,0,0,0,0,0 };
	std::vector<double> pq{ 0,0,0,0,0,0,1 };
	pes.push_back(pe);
	j["type"] = "321";
	j["pe"] = pes;
	j["pq"] = pqs;
	msg.copy(j.dump());

	client.socket().setOnLoseConnection(onLoseConnection);
	client.socket().setOnReceivedMsg(onReceivedMsg);


	while (true)
	{
		try
		{
			if (!client.socket().isConnected())
			{
				client.connect();
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
			client.socket().sendMsg(msg);

		}
		catch (const std::exception&)
		{
			client.connect();
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}

	}




	return 0;
}

