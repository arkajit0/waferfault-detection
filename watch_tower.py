import asyncio
# import nest_asyncio
from azure.eventhub.aio import EventHubConsumerClient
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore

Event_conn = "Endpoint=sb://eventwaffer.servicebus.windows.net/;SharedAccessKeyName=newpoliy;SharedAccessKey=wjLnIVyeFMfiggsm/StTrlm5w92hnTXrjGqReOI9l5Y=;EntityPath=wafferevent"
storage_conn = "DefaultEndpointsProtocol=https;AccountName=wafferfaultdetection;AccountKey=+g33VPCiyuZ96HTrw+dspAmKPGIGF/5GTyjO4nAsARgZ3rsrQhWfHocLUJkzYGErvW60168c1pCOYO1/zwfI9w==;EndpointSuffix=core.windows.net"


async def on_event(partition_context, event):
    # Print the event data.
    print(event)
    print("Received the event: \"{}\" from the partition with ID: \"{}\"".format(event.body_as_str(encoding='UTF-8'), partition_context.partition_id))

    # Update the checkpoint so that the program doesn't read the events
    # that it has already read when you run it next time.
    await partition_context.update_checkpoint(event)

async def main():
    # Create an Azure blob checkpoint store to store the checkpoints.
    checkpoint_store = BlobCheckpointStore.from_connection_string(storage_conn, "eventwatcher")

    # Create a consumer client for the event hub.
    client = EventHubConsumerClient.from_connection_string("Endpoint=sb://wafferfautdetection.servicebus.windows.net/;SharedAccessKeyName=newpolicy;SharedAccessKey=gSHIqAwpyxxkc0I+0T7p8nnd0UuiIRHjaO2JePxbFN8=;EntityPath=monitorfiles", consumer_group="$Default",
                                                           eventhub_name="wafferevent", checkpoint_store=checkpoint_store)
    async with client:
        # Call the receive method. Read from the beginning of the partition (starting_position: "-1")
        await client.receive(on_event=on_event,  starting_position="-1")

if __name__ == '__main__':
    # nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    # Run the main method.
    loop.run_until_complete(main())